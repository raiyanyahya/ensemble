"""Orchestrator + MCP tool, driven with fake providers (no network)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src import agent, coordinator, mcp_server, orchestrator
from src.models import PROVIDERS

MODELS = ["gpt4o", "claude", "deepseek"]


def _envvar(name: str) -> str:
    return PROVIDERS[name].env_var


def make_fake(name: str):
    async def fake(system: str, user: str, model_id: str) -> str:
        if "PHASE: proposing" in user:
            return f"## Proposal\nAnswer from {name}: 42.\n"
        if "PHASE: reviewing" in user:
            return f"## Reviews\n{name}: sound.\n"
        if "PHASE: rebuttal" in user:
            return f"## Rebuttal\n{name}: stands by it.\n"
        if "PHASE: voting" in user:
            return "## Vote\nFINALIZE: Participant A\n\n## Reasoning\nok.\n"
        if "PHASE: synthesis" in user:
            return f"## Synthesis\nMerged answer from {name}.\n"
        if "PHASE: confirm" in user:
            return "## Confirm\nAPPROVE\n"
        return ""
    return fake


@pytest.fixture
def council(monkeypatch, tmp_path):
    """Two+ providers with keys and stubbed calls, debates land in tmp_path."""
    for n in MODELS:
        monkeypatch.setenv(_envvar(n), "test-key")
    fake = {n: SimpleNamespace(call=make_fake(n)) for n in MODELS}
    monkeypatch.setattr(agent, "PROVIDERS", fake)
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)
    return tmp_path


def test_available_models_filters_by_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert orchestrator.available_models() == []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert orchestrator.available_models() == ["claude"]


def test_default_active_models_matches_canonical_order():
    # The DebateState default should reflect the canonical provider order so the
    # two never silently drift apart.
    from src.state import DebateState
    assert DebateState(debate_id="x", prompt="p").active_models == orchestrator.DEFAULT_ORDER


def test_build_debate_requires_two_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ValueError, match="at least 2"):
        orchestrator.build_debate("q")


def test_quick_mode_caps_rounds_at_one(council):
    state, _ = orchestrator.build_debate("q", quick=True, rounds=9)
    assert state.max_rounds == 1


def test_run_debate_reaches_consensus(council):
    state, debate_dir = orchestrator.build_debate("What is 6x7?", quick=True)
    status = asyncio.run(
        asyncio.wait_for(orchestrator.run_debate(state, debate_dir, poll_interval=0.01), timeout=15)
    )
    assert status == "done"
    assert "Answer from gpt4o" in (debate_dir / "final.md").read_text()


def test_vote_leaked_into_review_does_not_stall(monkeypatch, tmp_path):
    # A model that (against instructions) includes a "## Vote" section in its
    # REVIEW must not have that vote harvested early. Previously it did, which
    # made the model skip writing a real vote file, so the voting phase never
    # completed and the whole debate stalled into a deadlock.
    def leaky(name):
        async def fake(system: str, user: str, model_id: str) -> str:
            if "PHASE: proposing" in user:
                return f"## Proposal\nAnswer from {name}.\n"
            if "PHASE: reviewing" in user:
                return f"## Reviews\n{name}: sound.\n\n## Vote\nFINALIZE\n"  # leaked vote
            if "PHASE: rebuttal" in user:
                return f"## Rebuttal\n{name}: stands by it.\n"
            if "PHASE: voting" in user:
                return "## Vote\nFINALIZE: Participant A\n\n## Reasoning\nok.\n"
            if "PHASE: synthesis" in user:
                return f"## Synthesis\nMerged from {name}.\n"
            if "PHASE: confirm" in user:
                return "## Confirm\nAPPROVE\n"
            return ""
        return fake

    for n in MODELS:
        monkeypatch.setenv(_envvar(n), "test-key")
    fake = {n: SimpleNamespace(call=leaky(n)) for n in MODELS}
    monkeypatch.setattr(agent, "PROVIDERS", fake)
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)

    state, debate_dir = orchestrator.build_debate("q", quick=True)
    status = asyncio.run(
        asyncio.wait_for(
            orchestrator.run_debate(
                state, debate_dir, poll_interval=0.01, stall_timeout=5
            ),
            timeout=20,
        )
    )
    assert status == "done"


def _raiser(name):
    async def fake(system: str, user: str, model_id: str) -> str:
        raise RuntimeError(f"{name} provider is down")
    return fake


def _empty(name):
    async def fake(system: str, user: str, model_id: str) -> str:
        return ""  # writes a contribution file with no parseable section
    return fake


def _run(debate_dir, state, **kw):
    return asyncio.run(
        asyncio.wait_for(
            orchestrator.run_debate(state, debate_dir, poll_interval=0.01, **kw), timeout=20
        )
    )


def _setup(monkeypatch, tmp_path, **calls):
    """Wire fake provider ``call`` fns (keyed by model name) into a tmp debate dir."""
    for n in MODELS:
        monkeypatch.setenv(_envvar(n), "test-key")
    monkeypatch.setattr(agent, "PROVIDERS", {n: SimpleNamespace(call=calls[n]) for n in MODELS})
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)


def test_dead_provider_is_dropped_and_debate_completes(monkeypatch, tmp_path):
    # claude's API always fails; the debate should drop it and converge on the
    # other two rather than hang until the stall timeout.
    _setup(monkeypatch, tmp_path,
           gpt4o=make_fake("gpt4o"), deepseek=make_fake("deepseek"), claude=_raiser("claude"))

    state, debate_dir = orchestrator.build_debate("q", quick=True)
    status = _run(debate_dir, state, stall_timeout=10)

    assert status == "done"
    reloaded = coordinator.load_state(debate_dir / "state.json")
    assert "claude" in reloaded.dropped_models
    assert reloaded.active_models == ["gpt4o", "deepseek"]
    assert "Answer from gpt4o" in (debate_dir / "final.md").read_text()


def test_too_few_live_providers_deadlock_without_hang(monkeypatch, tmp_path):
    # Two of three providers fail; dropping both would leave one (< MIN_MODELS),
    # so the debate ends in a clean deadlock instead of hanging.
    _setup(monkeypatch, tmp_path,
           gpt4o=make_fake("gpt4o"), claude=_raiser("claude"), deepseek=_raiser("deepseek"))

    state, debate_dir = orchestrator.build_debate("q", quick=True)
    status = _run(debate_dir, state, stall_timeout=30)  # would hang 30s if broken

    assert status == "deadlocked"
    reloaded = coordinator.load_state(debate_dir / "state.json")
    assert {"claude", "deepseek"} <= set(reloaded.dropped_models)


def test_stall_salvages_partial_proposals(monkeypatch, tmp_path):
    # claude silently produces nothing (no error, so it isn't dropped) and the
    # proposing phase never completes. On the stall deadlock, the proposals the
    # other two did write must still be salvaged into final.md.
    _setup(monkeypatch, tmp_path,
           gpt4o=make_fake("gpt4o"), deepseek=make_fake("deepseek"), claude=_empty("claude"))

    state, debate_dir = orchestrator.build_debate("q", quick=True)
    status = _run(debate_dir, state, stall_timeout=3)

    assert status == "deadlocked"
    assert "Answer from gpt4o" in (debate_dir / "final.md").read_text()


def test_mcp_tool_returns_consensus(council):
    out = asyncio.run(
        asyncio.wait_for(
            mcp_server.run_ensemble_debate("Why?", quick=True, poll_interval=0.01), timeout=15
        )
    )
    assert "outcome=done" in out
    assert "Answer from claude" in out


def test_mcp_tool_errors_without_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)
    for n in MODELS:
        monkeypatch.delenv(_envvar(n), raising=False)
    out = asyncio.run(mcp_server.run_ensemble_debate("q"))
    assert out.startswith("ERROR")


def test_mcp_tool_is_registered():
    # The decorated tool is discoverable by the FastMCP server.
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert any(t.name == "ensemble_debate" for t in tools)
