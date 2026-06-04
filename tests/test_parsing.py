from src.coordinator import detect_vote, parse_output
from src.state import Phase, Vote


def test_parse_proposal_only():
    out = parse_output("## Proposal\nThe answer is 42.\n")
    assert out.proposal == "The answer is 42."
    assert out.reviews == ""
    assert out.vote is None


def test_parse_headerless_vote_directive():
    # Real-world stall trigger: the model put the bare directive as the first
    # line (as instructed) but omitted the "## Vote" header, then added
    # "## Reasoning". The vote is present and must be recovered, or the
    # coordinator waits forever for a vote it can't see.
    text = "FINALIZE: Participant B\n\n## Reasoning\nIts breakdown is clearest.\n"
    out = parse_output(text)
    assert out.vote == Vote.FINALIZE
    assert out.has_phase(Phase.VOTING)
    assert "Participant B" in out.vote_reasoning


def test_parse_headerless_revise_directive():
    out = parse_output("REVISE: tighten the proof\n\n## Reasoning\nNeeds a step.\n")
    assert out.vote == Vote.REVISE
    assert out.has_phase(Phase.VOTING)


def test_headerless_vote_in_proposal_is_stripped_for_proposing_phase():
    # A directive-looking line inside a *proposal* must not count as a vote when
    # the proposal phase harvests it (for_phase keeps only the proposal field).
    out = parse_output("## Proposal\nThe answer is 42.\nSPLIT the difference if unsure.\n")
    assert out.for_phase(Phase.PROPOSING).vote is None
    assert out.for_phase(Phase.PROPOSING).proposal.startswith("The answer is 42")


def test_parse_reviews_section():
    out = parse_output("## Reviews\nGPT is strong; DeepSeek missed X.\n")
    assert out.reviews.startswith("GPT is strong")
    assert out.proposal == ""


def test_parse_vote_and_reasoning():
    out = parse_output("## Vote\nFINALIZE\n\n## Reasoning\nGood enough.\n")
    assert out.vote == Vote.FINALIZE
    assert "FINALIZE" in out.vote_reasoning


def test_detect_vote_prefers_first_line():
    # Earlier code substring-matched FINALIZE first and got this wrong.
    assert detect_vote("REVISE: tighten the proof\nI considered FINALIZE but no.") == Vote.REVISE


def test_detect_vote_revise_with_focus():
    assert detect_vote("REVISE: focus on edge cases") == Vote.REVISE


def test_detect_vote_split():
    assert detect_vote("SPLIT: irreconcilable definitions") == Vote.SPLIT


def test_detect_vote_none_when_absent():
    assert detect_vote("I am not sure what to do here.") is None


def test_parse_rebuttal_section():
    out = parse_output("## Rebuttal\nMy proposal handles X via Y; conceding point Z.\n")
    assert out.rebuttal.startswith("My proposal handles X")
    assert out.proposal == "" and out.reviews == ""


def test_parse_proposal_preserves_internal_subheadings():
    # A model that structures its proposal with Markdown sub-headings must not
    # have everything after the first '##' silently dropped.
    text = (
        "## Proposal\n"
        "We should use Postgres.\n\n"
        "## Architecture\n"
        "Sharding matters here.\n\n"
        "## Tradeoffs\n"
        "Cost is the concern.\n\n"
        "## Vote\nFINALIZE\n"
    )
    out = parse_output(text)
    assert "Postgres" in out.proposal
    assert "Sharding matters here" in out.proposal
    assert "Cost is the concern" in out.proposal
    assert out.vote == Vote.FINALIZE


def test_parse_reviews_preserves_internal_subheadings():
    text = (
        "## Reviews\n"
        "Overall solid.\n\n"
        "## Strengths\n"
        "Clear reasoning.\n\n"
        "## Vote\nREVISE: tighten scope\n"
    )
    out = parse_output(text)
    assert "Overall solid" in out.reviews
    assert "Clear reasoning" in out.reviews
    assert out.vote == Vote.REVISE


def test_parse_multisection_full_output():
    text = (
        "## Proposal\nUse a hash map.\n\n"
        "## Reviews\nClaude's idea is cleaner.\n\n"
        "## Vote\nFINALIZE\n\n## Reasoning\nConverged.\n"
    )
    out = parse_output(text)
    assert "hash map" in out.proposal
    assert "cleaner" in out.reviews
    assert out.vote == Vote.FINALIZE


def test_parse_ranking_section():
    from src.coordinator import parse_output
    o = parse_output("## Vote\nFINALIZE: Participant A\n\n## Ranking\nB > C > A\n")
    assert o.vote is not None
    assert o.ranking == ["B", "C", "A"]


def test_parse_synthesis_and_confirm():
    from src.coordinator import parse_output
    assert parse_output("## Synthesis\nMerged answer here.\n").synthesis.startswith("Merged")
    assert parse_output("## Confirm\nAPPROVE\n").confirm == "APPROVE"
    assert parse_output("REJECT — the merge drops the minority view\n").confirm == "REJECT"
    assert parse_output("## Confirm\n(no directive)\n").confirm == ""
