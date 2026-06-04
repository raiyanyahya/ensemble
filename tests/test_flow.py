"""End-to-end debate flow with fake providers (no network).

This is the regression test for the original critical bug: proposals were lost
across phases and `final.md` came out empty.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src import agent, coordinator
from src.state import load_state

MODELS = ["gpt4o", "claude", "deepseek"]


def _all_finalize_a() -> dict[str, str]:
    # Every model endorses Participant A (the first model) so the round converges.
    return {n: "FINALIZE: Participant A" for n in MODELS}


def make_fake(name: str, vote_line: str = "FINALIZE", confirm: str = "APPROVE"):
    async def fake(system: str, user: str, model_id: str) -> str:
        if "PHASE: proposing" in user:
            return f"## Proposal\nAnswer from {name}: the result is 42.\n"
        if "PHASE: reviewing" in user:
            return f"## Reviews\n{name} notes the others are sound.\n"
        if "PHASE: rebuttal" in user:
            return f"## Rebuttal\n{name} stands by the proposal.\n"
        if "PHASE: voting" in user:
            return (f"## Vote\n{vote_line}\n\n## Ranking\nA > B > C\n"
                    f"\n## Reasoning\n{name} reasoning.\n")
        if "PHASE: synthesis" in user:
            return f"## Synthesis\nMerged answer from {name}; minority preserved.\n"
        if "PHASE: confirm" in user:
            return f"## Confirm\n{confirm}\n"
        return ""
    return fake


def _setup(monkeypatch, tmp_path, votes: dict[str, str], max_rounds: int = 5,
           confirms: dict[str, str] | None = None):
    confirms = confirms or {n: "APPROVE" for n in MODELS}
    for name in MODELS:
        monkeypatch.setenv(_envvar(name), "test-key")
    fake = {n: SimpleNamespace(call=make_fake(n, votes[n], confirms[n])) for n in MODELS}
    monkeypatch.setattr(agent, "PROVIDERS", fake)
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)

    state = coordinator.create_debate("What is 6 times 7?")
    state.active_models = list(MODELS)
    state.model_ids = {n: "fake-model" for n in MODELS}
    state.max_rounds = max_rounds
    debate_dir = coordinator.setup_debate_dir(state)
    return state, debate_dir


def _envvar(name: str) -> str:
    from src.models import PROVIDERS
    return PROVIDERS[name].env_var


async def _drive(debate_dir, active_models):
    agents = [
        asyncio.create_task(agent.agent_loop(n, debate_dir, poll_interval=0.01))
        for n in active_models
    ]
    status = await coordinator.coordinator_loop(debate_dir, poll_interval=0.01)
    for a in agents:
        a.cancel()
    await asyncio.gather(*agents, return_exceptions=True)
    return status


def test_full_debate_reaches_consensus_with_content(monkeypatch, tmp_path):
    state, debate_dir = _setup(monkeypatch, tmp_path, _all_finalize_a())

    status = asyncio.run(asyncio.wait_for(_drive(debate_dir, state.active_models), timeout=15))

    assert status == "done"
    final = (debate_dir / "final.md").read_text()
    # The regression assertion: every model's PROPOSAL text survived all phases.
    for n in MODELS:
        assert f"Answer from {n}" in final, f"{n}'s proposal missing from final.md"
    assert "endorsed" in final

    saved = load_state(debate_dir / "state.json")
    assert saved.status == "done"
    rs = saved.round_state
    for n in MODELS:
        assert rs.model_outputs[n].proposal  # accumulated, not clobbered
        assert rs.model_outputs[n].reviews
        assert rs.model_outputs[n].rebuttal
        assert rs.model_outputs[n].vote is not None


def test_phase_files_are_separate_artifacts(monkeypatch, tmp_path):
    state, debate_dir = _setup(monkeypatch, tmp_path, _all_finalize_a())
    asyncio.run(asyncio.wait_for(_drive(debate_dir, state.active_models), timeout=15))

    round1 = debate_dir / "round-001"
    for n in MODELS:
        assert (round1 / f"{n}.proposal.md").exists()
        assert (round1 / f"{n}.review.md").exists()
        assert (round1 / f"{n}.rebuttal.md").exists()
        assert (round1 / f"{n}.vote.md").exists()


def _advancing_fake(name: str):
    """Votes REVISE in round 1, then FINALIZE in round 2 — so the debate advances
    a round and then converges (no fixed cap needed)."""
    async def fake(system: str, user: str, model_id: str) -> str:
        if "PHASE: proposing" in user:
            return f"## Proposal\nAnswer from {name}: 42.\n"
        if "PHASE: reviewing" in user:
            return f"## Reviews\n{name}: ok.\n"
        if "PHASE: rebuttal" in user:
            return f"## Rebuttal\n{name}: ok.\n"
        if "PHASE: voting" in user:
            if "ROUND: 1/" in user:
                return "## Vote\nREVISE: sharpen it\n\n## Reasoning\nneeds work.\n"
            return "## Vote\nFINALIZE: Participant A\n\n## Reasoning\ngood now.\n"
        if "PHASE: synthesis" in user:
            return f"## Synthesis\nMerged answer from {name}.\n"
        if "PHASE: confirm" in user:
            return "## Confirm\nAPPROVE\n"
        return ""
    return fake


def test_stable_disagreement_ends_debate(monkeypatch, tmp_path):
    # Everyone keeps voting REVISE identically → positions never move → the debate
    # ends via stable-disagreement detection (not a round cap; fuse set high).
    state, debate_dir = _setup(
        monkeypatch, tmp_path, {n: "REVISE: more rigor" for n in MODELS}, max_rounds=50
    )
    status = asyncio.run(asyncio.wait_for(_drive(debate_dir, state.active_models), timeout=20))
    assert status == "deadlocked"
    saved = load_state(debate_dir / "state.json")
    assert saved.current_round == 2  # round 2 identical to round 1 → stop, far below fuse


def test_debate_advances_a_round_then_converges(monkeypatch, tmp_path):
    for name in MODELS:
        monkeypatch.setenv(_envvar(name), "test-key")
    fake = {n: SimpleNamespace(call=_advancing_fake(n)) for n in MODELS}
    monkeypatch.setattr(agent, "PROVIDERS", fake)
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)
    state = coordinator.create_debate("q")
    state.active_models = list(MODELS)
    state.model_ids = {n: "fake" for n in MODELS}
    state.max_rounds = 50
    debate_dir = coordinator.setup_debate_dir(state)

    status = asyncio.run(asyncio.wait_for(_drive(debate_dir, state.active_models), timeout=20))
    assert status == "done"
    saved = load_state(debate_dir / "state.json")
    assert saved.current_round == 2  # advanced past round 1, converged in round 2


@pytest.mark.parametrize("stall", [0.05])
def test_stall_timeout_declares_deadlock(monkeypatch, tmp_path, stall):
    # No agents at all → coordinator should give up via stall timeout.
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)
    state = coordinator.create_debate("nobody answers")
    state.active_models = list(MODELS)
    debate_dir = coordinator.setup_debate_dir(state)

    status = asyncio.run(
        asyncio.wait_for(
            coordinator.coordinator_loop(debate_dir, poll_interval=0.01, stall_timeout=stall),
            timeout=10,
        )
    )
    assert status == "deadlocked"
    assert (debate_dir / "final.md").exists()


def test_synthesis_used_when_confirmed(monkeypatch, tmp_path):
    state, debate_dir = _setup(monkeypatch, tmp_path, _all_finalize_a())
    status = asyncio.run(asyncio.wait_for(_drive(debate_dir, state.active_models), timeout=20))
    assert status == "done"
    saved = load_state(debate_dir / "state.json")
    rs = saved.round_state
    assert rs.synthesis_used is True
    assert rs.confirm_tally["APPROVE"] == len(MODELS)
    final = (debate_dir / "final.md").read_text()
    assert "Synthesis" in final and "minority preserved" in final
    assert "Ranking (Borda)" in final
    winner = rs.consensus_winner
    assert (debate_dir / "round-001" / f"{winner}.synthesis.md").exists()
    for n in MODELS:
        assert (debate_dir / "round-001" / f"{n}.confirm.md").exists()


def test_synthesis_rejected_falls_back_to_verbatim(monkeypatch, tmp_path):
    state, debate_dir = _setup(monkeypatch, tmp_path, _all_finalize_a(),
                               confirms={n: "REJECT" for n in MODELS})
    status = asyncio.run(asyncio.wait_for(_drive(debate_dir, state.active_models), timeout=20))
    assert status == "done"
    saved = load_state(debate_dir / "state.json")
    assert saved.round_state.synthesis_used is False
    final = (debate_dir / "final.md").read_text()
    for n in MODELS:
        assert f"Answer from {n}" in final
