"""Eval harness: grading logic (offline) + an end-to-end run with fake providers."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src import agent, coordinator
from src import eval as ev
from src.models import PROVIDERS

MODELS = ["gpt4o", "claude", "deepseek"]


def _envvar(name: str) -> str:
    return PROVIDERS[name].env_var


# --- grading -----------------------------------------------------------------


def test_grade_matches_standalone_token():
    assert ev.grade("The answer is 391.", ["391"])
    assert ev.grade("Canberra is the capital.", ["canberra"])


def test_grade_rejects_substring_of_larger_number():
    # "4" must not match "40" or "14".
    assert not ev.grade("The speed is 40 km/h", ["4"])
    assert not ev.grade("It happened in 2014", ["4"])


def test_grade_accepts_any_variant():
    assert ev.grade("about five cents", ["0.05", "5 cents", "five cents"])
    assert ev.grade("$0.05", ["0.05"])


def test_grade_false_when_absent():
    assert not ev.grade("I am not sure", ["391"])


# --- final-answer extraction (verbosity-bias fix) ---------------------------


def test_grade_uses_final_line_not_intermediate_numbers():
    # A verbose chain-of-thought mentions wrong intermediate numbers but
    # concludes correctly on the last line. Only the conclusion should count.
    text = (
        "First I tried 40, then I reconsidered and got 8, but those were wrong.\n"
        "The final answer is 360."
    )
    assert ev.grade(text, ["360"])


def test_grade_ignores_answer_buried_in_reasoning_middle():
    # The accepted token appears only in the discarded middle, while the
    # model's actual final answer is different — this must NOT score as correct.
    text = (
        "Some people guess the answer is 8.\n"
        "After working it through, the correct answer is 9."
    )
    assert not ev.grade(text, ["8"])


def test_grade_reads_labeled_answer_line():
    text = "Lots of reasoning here mentioning 12 and 99.\nAnswer: 42"
    assert ev.grade(text, ["42"])
    assert not ev.grade(text, ["12"])


def test_grade_yes_no_on_concluding_line():
    assert ev.grade("Let's see... \nTherefore, no.", ["no"])


def test_grade_fraction_answer():
    assert ev.grade("The probability is 1/2.", ["1/2", "0.5"])


# --- end-to-end evaluate with fakes -----------------------------------------


def _fake(name: str, answer: str):
    """A provider that answers `answer` to a direct single question, and plays a
    normal debate (proposing `answer`, then finalizing on Participant A)."""
    async def call(system: str, user: str, model_id: str) -> str:
        if "PHASE: proposing" in user:
            return f"## Proposal\nThe answer is {answer}.\n"
        if "PHASE: reviewing" in user:
            return f"## Reviews\n{name}: ok.\n"
        if "PHASE: rebuttal" in user:
            return f"## Rebuttal\n{name}: ok.\n"
        if "PHASE: voting" in user:
            return "## Vote\nFINALIZE: Participant A\n\n## Reasoning\nok.\n"
        if "PHASE: synthesis" in user:
            return f"## Synthesis\nThe answer is {answer}.\n"
        if "PHASE: confirm" in user:
            return "## Confirm\nAPPROVE\n"
        return f"The answer is {answer}."  # direct single-model query
    return call


def test_evaluate_scores_singles_and_ensemble(monkeypatch, tmp_path):
    for n in MODELS:
        monkeypatch.setenv(_envvar(n), "test-key")
    # gpt4o (Participant A) is right; the other two are wrong on the solo query.
    fakes = {
        "gpt4o": SimpleNamespace(call=_fake("gpt4o", "391")),
        "claude": SimpleNamespace(call=_fake("claude", "wrong")),
        "deepseek": SimpleNamespace(call=_fake("deepseek", "wrong")),
    }
    monkeypatch.setattr(ev, "PROVIDERS", fakes)
    monkeypatch.setattr(agent, "PROVIDERS", fakes)
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)

    questions = [{"question": "What is 17 times 23?", "accept": ["391"]}]
    rep = asyncio.run(ev.evaluate(questions, models=MODELS, poll_interval=0.01))

    assert rep.singles["gpt4o"].correct == 1
    assert rep.singles["claude"].correct == 0
    assert rep.singles["deepseek"].correct == 0
    # The debate endorses Participant A (gpt4o), whose proposal is "391" → correct.
    assert rep.ensemble.correct == 1
    outcomes = rep.rows[0]["outcomes"]
    assert outcomes["ensemble"] is True
    assert outcomes["gpt4o"] is True and outcomes["claude"] is False


def test_evaluate_baseline_and_log(monkeypatch, tmp_path):
    import json

    for n in MODELS:
        monkeypatch.setenv(_envvar(n), "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # sonnet baseline reuses it
    fakes = {
        "gpt4o": SimpleNamespace(call=_fake("gpt4o", "wrong")),
        "claude": SimpleNamespace(call=_fake("claude", "wrong")),
        "deepseek": SimpleNamespace(call=_fake("deepseek", "wrong")),
        "sonnet": SimpleNamespace(call=_fake("sonnet", "391")),  # strong model is right
    }
    monkeypatch.setattr(ev, "PROVIDERS", fakes)
    monkeypatch.setattr(agent, "PROVIDERS", fakes)
    monkeypatch.setattr(coordinator, "DEBATES_DIR", tmp_path)

    log = tmp_path / "run.jsonl"
    questions = [{"question": "What is 17 times 23?", "accept": ["391"], "category": "math"}]
    rep = asyncio.run(ev.evaluate(
        questions, models=MODELS, baseline="sonnet", poll_interval=0.01, log_path=log
    ))

    # Baseline tallied separately and scored correct.
    assert rep.baseline is not None and rep.baseline.correct == 1
    assert rep.baseline_name == "sonnet"
    # One JSONL record was written with the per-condition outcomes.
    record = json.loads(log.read_text().strip())
    assert record["outcomes"]["baseline:sonnet"] is True
    assert record["category"] == "math"
    assert "ensemble_status" in record
