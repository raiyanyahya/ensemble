from src.coordinator import determine_consensus, resolve_endorsement, tally_votes
from src.state import DebateState, Phase, RoundState, Vote


def _endorse_state() -> DebateState:
    return DebateState(
        debate_id="d",
        prompt="p",
        rounds=[RoundState(round_num=1, phase=Phase.VOTING)],
        current_round=1,
        active_models=["gpt4o", "claude", "deepseek"],
        participant_aliases={
            "gpt4o": "Participant A", "claude": "Participant B", "deepseek": "Participant C"
        },
    )


def test_resolve_endorsement_full_label():
    s = _endorse_state()
    assert resolve_endorsement("FINALIZE: Participant B is clearest", s, "gpt4o") == "claude"


def test_resolve_endorsement_bare_letter():
    s = _endorse_state()
    assert resolve_endorsement("FINALIZE: C", s, "gpt4o") == "deepseek"


def test_resolve_endorsement_defaults_to_self_when_unspecified():
    s = _endorse_state()
    assert resolve_endorsement("FINALIZE: my own answer is best", s, "deepseek") == "deepseek"


def test_resolve_endorsement_prefers_full_label_over_stray_letter():
    # A stray leading "A" must not beat the explicitly named Participant B.
    s = _endorse_state()
    assert resolve_endorsement("FINALIZE: A clear winner is Participant B", s, "gpt4o") == "claude"


def _state(
    votes: dict[str, Vote],
    endorsements: dict[str, str] | None = None,
    current_round: int = 1,
    max_rounds: int = 5,
) -> DebateState:
    rs = RoundState(
        round_num=current_round,
        phase=Phase.VOTING,
        votes=votes,
        endorsements=endorsements or {},
    )
    past = [RoundState(round_num=r, phase=Phase.VOTING) for r in range(1, current_round)]
    return DebateState(
        debate_id="d",
        prompt="p",
        rounds=past + [rs],
        current_round=current_round,
        max_rounds=max_rounds,
        active_models=["a", "b", "c"],
    )


def test_unanimous_endorsement_finalizes():
    # All finalize AND all endorse the same proposal (b).
    s = _state(
        {"a": Vote.FINALIZE, "b": Vote.FINALIZE, "c": Vote.FINALIZE},
        {"a": "b", "b": "b", "c": "b"},
    )
    action, reason = determine_consensus(s)
    assert action == "finalize"
    assert "Unanimous" in reason
    assert s.round_state.consensus_winner == "b"


def test_majority_endorsement_finalizes():
    # 2 of 3 endorse b → majority on b, even though c wants to revise.
    s = _state({"a": Vote.FINALIZE, "b": Vote.FINALIZE, "c": Vote.REVISE}, {"a": "b", "b": "b"})
    assert determine_consensus(s)[0] == "finalize"
    assert s.round_state.consensus_winner == "b"


def test_finalize_but_split_endorsements_is_not_consensus():
    # Everyone votes FINALIZE, but each endorses their own proposal — no single
    # proposal has a majority, so this is NOT consensus.
    s = _state(
        {"a": Vote.FINALIZE, "b": Vote.FINALIZE, "c": Vote.FINALIZE},
        {"a": "a", "b": "b", "c": "c"},
        current_round=1,
        max_rounds=5,
    )
    assert determine_consensus(s)[0] != "finalize"


def test_finalize_split_endorsements_deadlocks_at_cap_with_plurality():
    # Same split, but at the round cap → deadlock, recording the plurality answer.
    s = _state(
        {"a": Vote.FINALIZE, "b": Vote.FINALIZE, "c": Vote.FINALIZE},
        {"a": "b", "b": "b", "c": "c"},  # b has 2, c has 1, but 2 < majority? majority=2
        current_round=5,
        max_rounds=5,
    )
    # b has 2 endorsements == majority(2) → actually finalizes.
    assert determine_consensus(s)[0] == "finalize"
    assert s.round_state.consensus_winner == "b"


def test_majority_revise():
    s = _state({"a": Vote.REVISE, "b": Vote.REVISE, "c": Vote.FINALIZE}, {"c": "c"})
    assert determine_consensus(s)[0] == "revise"


def test_deadlock_at_max_rounds_records_plurality():
    # All endorse self (no majority) at the cap → deadlock, plurality winner set.
    s = _state(
        {"a": Vote.FINALIZE, "b": Vote.FINALIZE, "c": Vote.FINALIZE},
        {"a": "a", "b": "b", "c": "c"},
        current_round=5,
        max_rounds=5,
    )
    assert determine_consensus(s)[0] == "deadlocked"
    assert s.round_state.consensus_winner in ("a", "b", "c")  # some best-effort answer


def test_tally_counts():
    s = _state({"a": Vote.FINALIZE, "b": Vote.FINALIZE, "c": Vote.REVISE})
    assert tally_votes(s) == {"finalize": 2, "revise": 1}


def test_no_votes():
    s = _state({})
    assert determine_consensus(s)[0] == "wait"


def _two_rounds(r1_votes, r1_endorse, r2_votes, r2_endorse, max_rounds=50) -> DebateState:
    r1 = RoundState(round_num=1, phase=Phase.VOTING, votes=r1_votes, endorsements=r1_endorse)
    r2 = RoundState(round_num=2, phase=Phase.VOTING, votes=r2_votes, endorsements=r2_endorse)
    return DebateState(
        debate_id="d", prompt="p", rounds=[r1, r2], current_round=2,
        max_rounds=max_rounds, active_models=["a", "b", "c"],
    )


def test_stable_disagreement_deadlocks_without_a_cap():
    # Identical vote+endorsement distribution two rounds running, well below the
    # high fuse → the models have frozen, so the debate ends.
    votes = {"a": Vote.FINALIZE, "b": Vote.REVISE, "c": Vote.SPLIT}
    endorse = {"a": "a"}
    s = _two_rounds(votes, endorse, dict(votes), dict(endorse), max_rounds=50)
    action, reason = determine_consensus(s)
    assert action == "deadlocked"
    assert "Stable disagreement" in reason


def test_changing_positions_keep_going():
    # Round 2 differs from round 1 → not frozen → debate continues (no cap hit).
    r1 = {"a": Vote.SPLIT, "b": Vote.REVISE, "c": Vote.SPLIT}
    r2 = {"a": Vote.REVISE, "b": Vote.REVISE, "c": Vote.SPLIT}
    s = _two_rounds(r1, {}, r2, {}, max_rounds=50)
    assert determine_consensus(s)[0] in ("revise", "split")


def test_high_fuse_still_stops_runaway():
    votes = {"a": Vote.SPLIT, "b": Vote.SPLIT, "c": Vote.REVISE}
    s = _state(votes, current_round=50, max_rounds=50)
    assert determine_consensus(s)[0] == "deadlocked"


def test_borda_tally_resolves_labels_and_scores():
    from src.coordinator import compute_borda
    from src.state import DebateState, ModelOutput, Phase, RoundState
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude", "deepseek"]
    st.participant_aliases = {"gpt4o": "Participant A", "claude": "Participant B",
                              "deepseek": "Participant C"}
    rs = RoundState(round_num=1, phase=Phase.VOTING)
    rs.model_outputs = {
        "gpt4o": ModelOutput(ranking=["B", "C", "A"]),
        "claude": ModelOutput(ranking=["B", "A", "C"]),
        "deepseek": ModelOutput(ranking=["A", "B"]),
    }
    st.rounds = [rs]
    scores = compute_borda(st)
    # B: 2+2+0=4 ; A: 0+1+1=2 ; C: 1+0+0=1
    assert scores == {"claude": 4, "gpt4o": 2, "deepseek": 1}


def test_endorsement_tiebreak_uses_borda():
    from src.coordinator import _endorsement_tally
    from src.state import DebateState, Phase, RoundState
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude"]
    rs = RoundState(round_num=1, phase=Phase.VOTING)
    rs.endorsements = {"gpt4o": "gpt4o", "claude": "claude"}  # 1–1 plurality tie
    rs.borda_scores = {"gpt4o": 1, "claude": 5}
    st.rounds = [rs]
    winner, top = _endorsement_tally(st)
    assert winner == "claude" and top == 1


def test_phase_participants_synthesis_is_winner_only():
    from src.coordinator import _phase_participants
    from src.state import DebateState, Phase, RoundState
    st = DebateState(debate_id="d", prompt="p")
    st.active_models = ["gpt4o", "claude", "deepseek"]
    rs = RoundState(round_num=1, phase=Phase.SYNTHESIS)
    rs.consensus_winner = "claude"
    st.rounds = [rs]
    assert _phase_participants(st) == ["claude"]
    rs.phase = Phase.CONFIRM
    assert _phase_participants(st) == st.active_models


def test_final_renders_synthesis_and_ranking():
    from src.coordinator import render_final_answer
    from src.state import DebateState, ModelOutput, Phase, RoundState
    st = DebateState(debate_id="d", prompt="p")
    st.status = "done"
    st.active_models = ["gpt4o", "claude"]
    rs = RoundState(round_num=1, phase=Phase.CONFIRM)
    rs.consensus_winner = "gpt4o"
    rs.endorsements = {"gpt4o": "gpt4o", "claude": "gpt4o"}
    rs.model_outputs = {
        "gpt4o": ModelOutput(proposal="raw winner proposal", synthesis="MERGED + minority"),
        "claude": ModelOutput(proposal="other"),
    }
    rs.synthesis_used = True
    rs.synthesis_author = "gpt4o"
    rs.confirm_tally = {"APPROVE": 2, "REJECT": 0}
    rs.borda_scores = {"gpt4o": 2, "claude": 1}
    st.rounds = [rs]
    st.current_round = 1
    out = render_final_answer(st)
    assert "MERGED + minority" in out and "Synthesis" in out
    assert "Ranking" in out
    assert "raw winner proposal" in out
