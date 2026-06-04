"""Evaluation harness: does the ensemble debate actually beat a single model?

For each question (with a known short answer) it runs every available model
*solo* (one call) and the *ensemble* (a quick debate), grades each answer by
normalized token matching, and reports accuracy + cost per condition. The point
is to put numbers on the central claim rather than assert it.

Run:  ensemble-eval            (after `pip install -e .`)
      ensemble-eval --limit 5  (quick smoke run)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .coordinator import load_state
from .models import PROVIDERS, default_model_id, get_api_key, usage_cost
from .orchestrator import available_models, build_debate, run_debate

# Objective questions with short, checkable answers. A few are deliberately
# "LLM traps" (bat-and-ball, letter counting) where a single cheap model often
# slips — the cases where a second/third opinion should help, if it helps at all.
QUESTIONS: list[dict] = [
    {"question": "What is 17 multiplied by 23?", "accept": ["391"]},
    {"question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
                 "than the ball. How much does the ball cost, in dollars?",
     "accept": ["0.05", ".05", "5 cents", "5 cent", "five cents"]},
    {"question": "How many letter 's' characters are in the word 'necessary'?",
     "accept": ["2", "two"]},
    {"question": "If a train travels 60 km in 1.5 hours, what is its average speed "
                 "in km/h?", "accept": ["40"]},
    {"question": "What is the capital city of Australia?", "accept": ["canberra"]},
    {"question": "What is the chemical symbol for gold?", "accept": ["au"]},
    {"question": "In what year did the Berlin Wall fall?", "accept": ["1989"]},
    {"question": "What is 15% of 200?", "accept": ["30"]},
    {"question": "What is the square root of 144?", "accept": ["12", "twelve"]},
    {"question": "Which planet is known as the Red Planet?", "accept": ["mars"]},
    {"question": "What is the next number in the sequence 2, 4, 8, 16, ...?",
     "accept": ["32"]},
    {"question": "How many degrees are in the interior angles of a triangle, in "
                 "total?", "accept": ["180"]},
    {"question": "What is the smallest prime number?", "accept": ["2", "two"]},
    {"question": "If you have 12 apples and give away a third, how many remain?",
     "accept": ["8", "eight"]},
    {"question": "What is the boiling point of water at sea level in degrees "
                 "Celsius?", "accept": ["100"]},
]

_SYSTEM = "You are a careful expert. Answer the question correctly and as concisely as possible."


def normalize(s: str) -> str:
    """Lowercase, keep letters/digits/dots/spaces, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9. ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Only an *anchored* label line counts ("Answer: 42"), not a passing mention of
# the word "answer" inside reasoning ("...the answer is 8, but actually...").
_ANSWER_LABEL = re.compile(r"(?i)^(?:final\s+answer|answer)\b\s*[:\-]?\s*(.+)$")


def final_segments(text: str) -> list[str]:
    """The short region(s) where a model's *final* answer lives.

    To remove the verbosity bias (a longer output is more likely to mention an
    accepted token somewhere in its reasoning), grading looks only at the
    concluding line plus any explicit "Answer: ..." line — never the discarded
    middle. This puts a concise single model and a verbose ensemble proposal on
    equal footing.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    segments = [lines[-1]]
    for ln in lines:
        m = _ANSWER_LABEL.search(ln)
        if m:
            segments.append(m.group(1))
    return segments


def grade(text: str, accept: list[str]) -> bool:
    """True if any accepted answer is a standalone token in the final segment(s).

    Token-boundary matching keeps "4" from matching "14" or "40"; restricting
    the search to :func:`final_segments` keeps intermediate reasoning numbers
    from counting as the answer.
    """
    segs = [normalize(s) for s in final_segments(text)]
    for a in accept:
        na = normalize(a)
        if not na:
            continue
        if any(re.search(rf"(?<!\w){re.escape(na)}(?!\w)", s) for s in segs):
            return True
    return False


async def single_answer(model: str, question: str) -> tuple[str, float]:
    """Ask one model directly. Returns (answer_text, estimated_cost)."""
    model_id = default_model_id(model)
    result = await PROVIDERS[model].call(_SYSTEM, f"{question}\nAnswer concisely.", model_id)
    text = result if isinstance(result, str) else result.text
    usage = None if isinstance(result, str) else result.usage
    cost = (usage_cost(model, model_id, usage) or 0.0) if usage else 0.0
    return text, cost


async def ensemble_answer(
    question: str, models: list[str] | None = None, poll_interval: float | None = None,
    stall_timeout: float = 60.0,
) -> tuple[str, float, str, str]:
    """Run a quick debate. Returns (answer_text, cost, status, reason)."""
    state, debate_dir = build_debate(question, models=models, quick=True)
    await run_debate(state, debate_dir, stall_timeout=stall_timeout, poll_interval=poll_interval)
    final = load_state(debate_dir / "state.json")
    rs = final.rounds[final.current_round - 1]
    out = rs.model_outputs.get(rs.consensus_winner) if rs.consensus_winner else None
    text = (out.proposal if out and out.proposal else final.final_answer) or ""
    return text, final.total_cost(), final.status, rs.consensus_reason or ""


@dataclass
class Tally:
    correct: int = 0
    total: int = 0
    cost: float = 0.0

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0


@dataclass
class Report:
    singles: dict[str, Tally] = field(default_factory=dict)
    ensemble: Tally = field(default_factory=Tally)
    baseline: Tally | None = None
    baseline_name: str = ""
    rows: list[dict] = field(default_factory=list)
    # Debates that did not reach clean consensus (status != "done"): a mix of
    # genuine deadlocks and stall-timeouts. The per-question log keeps the status
    # and reason so the two can be told apart on audit.
    ensemble_nonconsensus: int = 0

    def best_single(self) -> tuple[str, Tally] | None:
        return max(self.singles.items(), key=lambda kv: kv[1].correct, default=None)


def _record(row: dict, accept: list[str], label: str, tally: Tally, text: str, cost: float) -> bool:
    """Grade one condition's answer, fold it into ``tally``, and log it on ``row``."""
    ok = grade(text, accept)
    tally.total += 1
    tally.correct += int(ok)
    tally.cost += cost
    row["outcomes"][label] = ok
    row["answers"][label] = text.strip()[:500]
    row["costs"][label] = round(cost, 6)
    return ok


async def evaluate(
    questions: list[dict], models: list[str] | None = None, poll_interval: float | None = None,
    stall_timeout: float = 60.0, baseline: str | None = None, delay: float = 0.0,
    log_path: Path | None = None,
) -> Report:
    """Run every condition over ``questions`` and tally accuracy + cost.

    ``baseline`` names a strong single model (provider key, e.g. ``"sonnet"``)
    run once per question as the headline comparison. ``delay`` sleeps between
    questions to avoid rate-limit bursts. ``log_path`` appends one JSONL record
    per question (per-condition outcome, answer text, cost) for later auditing.
    """
    models = models or available_models()
    rep = Report(
        singles={m: Tally() for m in models},
        baseline=Tally() if baseline else None,
        baseline_name=baseline or "",
    )
    if log_path:
        log_path.write_text("")  # truncate any previous run

    for i, q in enumerate(questions):
        question, accept = q["question"], q["accept"]
        row: dict = {"i": i, "question": question, "category": q.get("category", ""),
                     "accept": accept, "outcomes": {}, "answers": {}, "costs": {}}

        for m in models:
            text, cost = await single_answer(m, question)
            _record(row, accept, m, rep.singles[m], text, cost)

        etext, ecost, estatus, ereason = await ensemble_answer(
            question, models=models, poll_interval=poll_interval, stall_timeout=stall_timeout
        )
        _record(row, accept, "ensemble", rep.ensemble, etext, ecost)
        row["ensemble_status"] = estatus
        row["ensemble_reason"] = ereason
        if estatus != "done":
            rep.ensemble_nonconsensus += 1

        if baseline and rep.baseline is not None:
            btext, bcost = await single_answer(baseline, question)
            _record(row, accept, f"baseline:{baseline}", rep.baseline, btext, bcost)

        rep.rows.append(row)
        if log_path:
            with log_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        if delay and i < len(questions) - 1:
            await asyncio.sleep(delay)
    return rep


def _row(label: str, t: Tally) -> str:
    """One report line: score, accuracy, total cost, and cost per correct answer."""
    cpc = f"${t.cost / t.correct:>8.4f}" if t.correct else f"{'n/a':>9}"
    return f"{label:<18}{t.correct}/{t.total:<6}{t.pct:>8.1f}%  ${t.cost:>8.4f}  {cpc}"


def format_report(rep: Report) -> str:
    width = 60
    lines = ["", f"{'Condition':<18}{'Score':>8}{'Accuracy':>10}{'Cost':>11}{'$/correct':>11}",
             "-" * width]
    for m, t in rep.singles.items():
        lines.append(_row(m, t))
    if rep.baseline is not None:
        lines.append("-" * width)
        lines.append(_row(f"BASELINE ({rep.baseline_name})", rep.baseline))
    lines.append("-" * width)
    e = rep.ensemble
    lines.append(_row("ENSEMBLE", e))

    def verdict(label: str, other: Tally) -> str:
        d = e.pct - other.pct
        rel = "beat" if d > 0 else "did NOT beat" if d < 0 else "tied"
        cost = (f" at {e.cost / other.cost:.1f}x the cost" if other.cost else "")
        return f"Ensemble {rel} {label} (Δ {d:+.1f} pts){cost}."

    best = rep.best_single()
    tail: list[str] = [""]
    if best:
        tail.append(verdict(f"the best single cheap model ({best[0]})", best[1]))
    if rep.baseline is not None:
        tail.append("HEADLINE: " + verdict(f"the strong baseline ({rep.baseline_name})",
                                           rep.baseline))
    if rep.ensemble_nonconsensus:
        tail.append(f"Reliability: {rep.ensemble_nonconsensus}/{e.total} debates ended "
                    "without clean consensus (see the per-question log).")
    return "\n".join(lines + tail)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ensemble vs single models.")
    ap.add_argument("--limit", type=int, default=0, help="Only run the first N questions")
    ap.add_argument("--models", default="", help="Comma-separated subset, e.g. gpt4o,claude")
    ap.add_argument("--dataset", type=Path,
                    help="JSONL file of {question, accept} (overrides built-in)")
    ap.add_argument("--baseline", default="",
                    help="Strong single-model baseline to compare against, e.g. 'sonnet'")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="Seconds to sleep between questions (eases rate-limit bursts)")
    ap.add_argument("--stall-timeout", type=float, default=60.0,
                    help="Seconds a debate may stall before the coordinator gives up")
    ap.add_argument("--log", type=Path,
                    help="Write a per-question JSONL log here for auditing")
    args = ap.parse_args()

    questions = QUESTIONS
    if args.dataset:
        questions = [
            json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()
        ]
    if args.limit:
        questions = questions[: args.limit]

    requested = [m.strip() for m in args.models.split(",") if m.strip()] or None
    models = available_models(requested)
    if len(models) < 2:
        raise SystemExit("Need at least 2 providers with API keys set (and ideally all 3).")

    baseline = args.baseline.strip() or None
    if baseline and not get_api_key(baseline):
        raise SystemExit(f"--baseline {baseline!r} has no API key set (env var missing).")

    extra = f"; baseline={baseline}" if baseline else ""
    print(f"Evaluating {len(questions)} questions across: {', '.join(models)}{extra}")
    rep = asyncio.run(evaluate(
        questions, models=models, baseline=baseline, delay=args.delay,
        stall_timeout=args.stall_timeout, log_path=args.log,
    ))
    print(format_report(rep))
    if args.log:
        print(f"\nPer-question log written to {args.log}")


if __name__ == "__main__":
    main()
