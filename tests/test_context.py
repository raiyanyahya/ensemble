"""Context assembly: the current round is shown to peers in full (no mid-sentence
truncation), and past rounds use generous caps."""
from __future__ import annotations

from src import agent
from src.state import DebateState, ModelOutput, Phase, RoundState, assign_aliases

ACTIVE = ["gpt4o", "claude", "deepseek"]


def _state(phase: Phase, current_outputs=None, past=None) -> DebateState:
    past = past or []
    rounds = past + [RoundState(round_num=len(past) + 1, phase=phase,
                                model_outputs=current_outputs or {})]
    return DebateState(
        debate_id="d", prompt="q", rounds=rounds, current_round=len(rounds),
        active_models=ACTIVE, participant_aliases=assign_aliases(ACTIVE),
    )


def test_current_round_proposal_shown_in_full():
    big = "X" * 5000  # well beyond the old 2000-char cap
    state = _state(Phase.REVIEWING, current_outputs={"claude": ModelOutput(proposal=big)})
    _, user = agent.build_agent_prompt("gpt4o", state)
    assert big in user, "current-round proposal was truncated"


def test_past_round_proposal_cap_is_generous():
    big = "Y" * 4000  # exceeds the old 1500-char past cap
    past = [RoundState(round_num=1, phase=Phase.VOTING,
                       model_outputs={"claude": ModelOutput(proposal=big)})]
    state = _state(Phase.PROPOSING, past=past)
    _, user = agent.build_agent_prompt("gpt4o", state)
    assert big in user, "past-round proposal cap is still too small"
