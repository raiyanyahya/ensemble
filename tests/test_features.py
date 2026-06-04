"""Cost/token tracking, budgets, prompt-cache usage, grounding, and roles."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src import agent, coordinator, models, orchestrator, search
from src.models import ModelResult, Usage, usage_cost
from src.state import DebateState, ModelUsage, Phase, RoundState, Source

MODELS = ["gpt4o", "claude", "deepseek"]


def _envvar(name: str) -> str:
    return models.PROVIDERS[name].env_var


def make_fake(name: str, usage: Usage):
    async def fake(system: str, user: str, model_id: str) -> ModelResult:
        if "PHASE: proposing" in user:
            text = f"## Proposal\n{name}: 42 [1].\n"
        elif "PHASE: reviewing" in user:
            text = f"## Reviews\n{name}: agree.\n"
        elif "PHASE: rebuttal" in user:
            text = f"## Rebuttal\n{name}: stands.\n"
        elif "PHASE: synthesis" in user:
            text = f"## Synthesis\n{name}: merged.\n"
        elif "PHASE: confirm" in user:
            text = "## Confirm\nAPPROVE\n"
        else:
            text = "## Vote\nFINALIZE: Participant A\n\n## Reasoning\nok.\n"
        return ModelResult(text, usage)
    return fake


@pytest.fixture
def council(monkeypatch, tmp_path):
    for n in MODELS:
        monkeypatch.setenv(_envvar(n), "k")
    fake = {n: SimpleNamespace(call=make_fake(n, Usage(1000, 500, 200))) for n in MODELS}
    monkeypatch.setattr(agent, "PROVIDERS", fake)
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)
    return tmp_path


# --- cost / usage ------------------------------------------------------------


def test_usage_cost_known_model():
    # gpt-4o-mini: 800 non-cached in @ .15 + 200 cached @ .075 + 500 out @ .60, per 1M
    cost = usage_cost("gpt4o", "gpt-4o-mini", Usage(1000, 500, 200))
    expected = (800 * 0.15 + 200 * 0.15 * 0.5 + 500 * 0.60) / 1_000_000
    assert cost == pytest.approx(expected)


def test_anthropic_cache_creation_billed_at_write_rate():
    # Anthropic bills cache *writes* at a premium over base input (1.25x), while
    # cache *reads* are discounted. 1000 total input = 600 fresh + 300 read +
    # 100 creation; claude haiku is 1.00 in / 5.00 out per 1M, read discount 0.1.
    usage = Usage(
        input_tokens=1000, output_tokens=200, cached_tokens=300, cache_creation_tokens=100
    )
    cost = usage_cost("claude", "claude-haiku-4-5-20251001", usage)
    expected = (
        600 * 1.00            # fresh input
        + 300 * 1.00 * 0.1    # cache reads
        + 100 * 1.00 * 1.25   # cache writes (premium)
        + 200 * 5.00          # output
    ) / 1_000_000
    assert cost == pytest.approx(expected)


def test_usage_cost_unknown_model_returns_none():
    assert usage_cost("gpt4o", "some-openrouter/model", Usage(100, 100, 0)) is None


def test_model_usage_marks_unknown():
    u = ModelUsage()
    u.add(100, 50, 0, 0.01)
    u.add(100, 50, 0, None)  # unknown pricing
    assert u.calls == 2
    assert u.cost == pytest.approx(0.01)
    assert u.cost_known is False


def test_debate_tracks_cost_in_final(council):
    state, debate_dir = orchestrator.build_debate("What is 6x7?", quick=True)
    status = asyncio.run(
        asyncio.wait_for(orchestrator.run_debate(state, debate_dir, poll_interval=0.01), timeout=15)
    )
    assert status == "done"
    reloaded = coordinator.load_state(debate_dir / "state.json")
    assert reloaded.total_cost() > 0
    # propose + review + rebuttal + vote + confirm = 5; the endorsed winner
    # (Participant A = gpt4o) also authors the synthesis, so 6.
    for n in MODELS:
        expected = 6 if n == "gpt4o" else 5
        assert reloaded.usage[n].calls == expected
        assert reloaded.usage[n].cached_tokens == 200 * expected  # 200 per call
    assert "## Cost" in (debate_dir / "final.md").read_text()


def test_budget_stops_debate(council):
    # Each call ~$0.00045 (gpt4o); a tiny budget trips after the first phase.
    state, debate_dir = orchestrator.build_debate("q", quick=True, budget=0.0001)
    status = asyncio.run(
        asyncio.wait_for(orchestrator.run_debate(state, debate_dir, poll_interval=0.01), timeout=15)
    )
    assert status == "over_budget"
    reloaded = coordinator.load_state(debate_dir / "state.json")
    assert reloaded.is_finished
    assert "budget" in (debate_dir / "final.md").read_text().lower()


# --- parsing usage out of provider responses ---------------------------------


def test_openai_usage_parsed(monkeypatch):
    async def fake_post(url, **kw):
        return {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30,
                      "prompt_tokens_details": {"cached_tokens": 40}},
        }
    monkeypatch.setattr(models, "_post_json", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    r = asyncio.run(models.call_openai("s", "u", "gpt-4o-mini"))
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.cached_tokens) == (120, 30, 40)


def test_anthropic_usage_includes_cache_reads(monkeypatch):
    async def fake_post(url, **kw):
        return {
            "content": [{"text": "hi"}],
            "usage": {"input_tokens": 10, "output_tokens": 5,
                      "cache_read_input_tokens": 90, "cache_creation_input_tokens": 0},
        }
    monkeypatch.setattr(models, "_post_json", fake_post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    r = asyncio.run(models.call_anthropic("s", "u", "claude-haiku-4-5-20251001"))
    assert r.usage.input_tokens == 100  # 10 + 90 cache read
    assert r.usage.cached_tokens == 90


# --- grounding ---------------------------------------------------------------


def test_web_search_disabled_without_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert search.search_enabled() is False
    assert asyncio.run(search.web_search("anything")) == []


def test_ground_debate_populates_sources(council, monkeypatch):
    async def fake_search(query, **kw):
        return [Source(title="T", url="http://x", snippet="snip")]
    monkeypatch.setattr(orchestrator, "web_search", fake_search)

    state, debate_dir = orchestrator.build_debate("q", quick=True, grounding=True)
    asyncio.run(orchestrator.ground_debate(state, debate_dir))
    assert len(state.sources) == 1
    assert state.sources[0].url == "http://x"


def test_sources_injected_into_prompt():
    state = DebateState(
        debate_id="d", prompt="p", active_models=["gpt4o", "claude"],
        rounds=[RoundState(round_num=1, phase=Phase.PROPOSING)], current_round=1,
        sources=[Source(title="Doc", url="http://x", snippet="snip")],
    )
    _, user = agent.build_agent_prompt("gpt4o", state)
    assert "GROUNDING SOURCES" in user
    assert "http://x" in user
    assert "cite" in user.lower()


# --- roles -------------------------------------------------------------------


def test_assign_roles_diverse():
    roles = orchestrator.assign_roles(["gpt4o", "claude", "deepseek"], "diverse")
    assert "skeptical" in roles["gpt4o"].lower()
    assert "optimistic" in roles["claude"].lower()


def test_assign_roles_override_custom_text():
    roles = orchestrator.assign_roles(["gpt4o"], "none", {"gpt4o": "argue like a security auditor"})
    assert roles["gpt4o"] == "argue like a security auditor"


def test_role_injected_into_prompt():
    state = DebateState(
        debate_id="d", prompt="p", active_models=["gpt4o"],
        rounds=[RoundState(round_num=1, phase=Phase.PROPOSING)], current_round=1,
        roles={"gpt4o": "Be the skeptic."},
    )
    _, user = agent.build_agent_prompt("gpt4o", state)
    assert "ASSIGNED STANCE" in user
    assert "Be the skeptic." in user


def test_agent_participation_and_prompt_phases():
    from src.agent import _agent_participates, build_agent_prompt
    from src.state import DebateState, Phase, RoundState
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude"]
    rs = RoundState(round_num=1, phase=Phase.SYNTHESIS)
    rs.consensus_winner = "gpt4o"
    st.rounds = [rs]
    st.current_round = 1
    st.current_phase = Phase.SYNTHESIS
    assert _agent_participates("gpt4o", st) is True
    assert _agent_participates("claude", st) is False
    rs.phase = Phase.CONFIRM
    st.current_phase = Phase.CONFIRM
    assert _agent_participates("claude", st) is True
    _, user = build_agent_prompt("claude", st)
    assert "CONFIRM" in user


def _state_with_round(phase, **rs_kwargs):
    from src.state import DebateState, RoundState
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude", "deepseek"]
    rs = RoundState(round_num=1, phase=phase, **rs_kwargs)
    st.rounds = [rs]
    st.current_round = 1
    st.current_phase = phase
    return st


def test_phase_footer_hidden_before_vote():
    from src.cli import _phase_footer
    from src.state import Phase
    assert _phase_footer(_state_with_round(Phase.PROPOSING)) is None


def test_phase_footer_shows_synthesis_confirm_and_ranking():
    from src.cli import _phase_footer
    from src.state import ModelOutput, Phase
    st = _state_with_round(
        Phase.CONFIRM,
        consensus_winner="claude",
        borda_scores={"gpt4o": 0, "claude": 5, "deepseek": 4},
        confirm_tally={"APPROVE": 3, "REJECT": 0},
        synthesis_used=True,
    )
    st.round_state.model_outputs = {"claude": ModelOutput(synthesis="merged")}
    st.status = "done"
    out = _phase_footer(st)
    assert out is not None
    assert "ranking" in out and "synthesis" in out and "confirm" in out
    assert "claude" in out                 # winner + borda leader
    assert "3/3" in out and "adopted" in out


def test_phase_footer_verbatim_when_rejected():
    from src.cli import _phase_footer
    from src.state import ModelOutput, Phase
    st = _state_with_round(
        Phase.CONFIRM,
        consensus_winner="claude",
        confirm_tally={"APPROVE": 1, "REJECT": 2},
        synthesis_used=False,
    )
    st.round_state.model_outputs = {"claude": ModelOutput(synthesis="merged")}
    st.status = "done"
    out = _phase_footer(st)
    assert "verbatim winner" in out
