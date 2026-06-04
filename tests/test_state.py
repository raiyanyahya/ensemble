from src.coordinator import create_debate
from src.state import (
    DebateState,
    ModelOutput,
    Phase,
    Vote,
    load_state,
    save_state,
)


def test_merge_accumulates_across_phases():
    """The data-loss bug: contributions from different phases must accumulate."""
    acc = ModelOutput()
    acc.merge_from(ModelOutput(proposal="answer 42"))
    acc.merge_from(ModelOutput(reviews="others are fine"))
    acc.merge_from(ModelOutput(vote=Vote.FINALIZE, vote_reasoning="FINALIZE"))
    assert acc.proposal == "answer 42"  # survived all three merges
    assert acc.reviews == "others are fine"
    assert acc.vote == Vote.FINALIZE


def test_merge_does_not_clobber_with_empty():
    acc = ModelOutput(proposal="keep me")
    acc.merge_from(ModelOutput(reviews="new"))
    assert acc.proposal == "keep me"


def test_has_phase():
    o = ModelOutput(proposal="x")
    assert o.has_phase(Phase.PROPOSING)
    assert not o.has_phase(Phase.REVIEWING)
    assert not o.has_phase(Phase.VOTING)


def test_majority():
    s = DebateState(debate_id="d", prompt="p", active_models=["a", "b", "c"])
    assert s.majority() == 2
    s.active_models = ["a", "b"]
    assert s.majority() == 2


def test_atomic_save_round_trip(tmp_path):
    state = create_debate("test prompt")
    p = tmp_path / "state.json"
    save_state(state, p)
    loaded = load_state(p)
    assert loaded.debate_id == state.debate_id
    assert loaded.prompt == "test prompt"
    assert loaded.updated_at  # stamped on save
    assert not (tmp_path / ".state-").exists()  # no temp leftovers


def test_save_leaves_no_temp_files(tmp_path):
    state = create_debate("x")
    p = tmp_path / "state.json"
    save_state(state, p)
    leftovers = [f for f in tmp_path.iterdir() if f.name.startswith(".state-")]
    assert leftovers == []


def test_new_phases_and_output_fields():
    from src.state import PHASE_FILE_SUFFIX
    assert Phase.SYNTHESIS == "synthesis"
    assert Phase.CONFIRM == "confirm"
    assert PHASE_FILE_SUFFIX[Phase.SYNTHESIS] == "synthesis"
    assert PHASE_FILE_SUFFIX[Phase.CONFIRM] == "confirm"

    o = ModelOutput(synthesis="merged", confirm="APPROVE", ranking=["B", "A"])
    assert o.has_phase(Phase.SYNTHESIS)
    assert o.has_phase(Phase.CONFIRM)
    # ranking rides along the vote phase, doesn't gate it
    assert ModelOutput(vote=None, ranking=["A"]).has_phase(Phase.VOTING) is False

    merged = ModelOutput()
    merged.merge_from(o)
    assert merged.synthesis == "merged" and merged.confirm == "APPROVE"
    assert merged.ranking == ["B", "A"]
    # for_phase isolates the field a phase owns
    assert ModelOutput(synthesis="x", confirm="y").for_phase(Phase.SYNTHESIS).confirm == ""
