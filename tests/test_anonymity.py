"""The inter-model prompts must not reveal that peers are specific commercial
models — participants are anonymized so they judge arguments on merit."""
from __future__ import annotations

from src import agent, coordinator, orchestrator
from src.state import DebateState, ModelOutput, Phase, RoundState, assign_aliases

MODELS = ["gpt4o", "claude", "deepseek"]
BRANDS = ["OpenAI", "Anthropic", "DeepSeek", "GPT", "Claude", "gpt4o", "claude", "deepseek"]


def test_build_debate_assigns_aliases(monkeypatch, tmp_path):
    for n in MODELS:
        monkeypatch.setenv({"gpt4o": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY",
                            "deepseek": "DEEPSEEK_API_KEY"}[n], "k")
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)
    state, _ = orchestrator.build_debate("q", quick=True)
    assert set(state.participant_aliases) == set(state.active_models)
    assert sorted(state.participant_aliases.values()) == [
        "Participant A", "Participant B", "Participant C"
    ]


def test_agent_prompt_hides_provider_identities():
    active = MODELS
    rs = RoundState(
        round_num=1,
        phase=Phase.REVIEWING,
        model_outputs={
            "claude": ModelOutput(proposal="Idea one."),
            "deepseek": ModelOutput(proposal="Idea two."),
        },
    )
    state = DebateState(
        debate_id="20260101-000000-abc123",
        prompt="should we ship on friday?",
        rounds=[rs],
        current_round=1,
        active_models=active,
        participant_aliases=assign_aliases(active),
    )
    system, user = agent.build_agent_prompt("gpt4o", state)
    blob = system + "\n" + user
    for brand in BRANDS:
        assert brand not in blob, f"leaked identity {brand!r} into the prompt"
    assert "Participant" in blob
    # peers' content still reaches the model, just unattributed-by-brand
    assert "Idea one." in blob and "Idea two." in blob
