# Next eval steps — plan for 2026-06-05

_Owner: raiyan. Follows `harder-eval-plan.md` (executed 2026-06-04)._

## Where we are (recap)

- Harness shipped: `harder.jsonl` (72 verified Qs), strong-model baseline (Sonnet),
  per-question JSONL logging, reliability knobs, final-answer grading.
- **Stall bug found + fixed:** 45/138 vote files held a valid directive with no
  `## Vote` header; the parser dropped them → coordinator timed out. `parse_output`
  now recovers unwrapped directives (regression tests in `test_parsing.py`).
- The recorded 72-Q run (`evals/harder-run.jsonl`) is **pre-fix** — 41/72 debates
  stalled, so its 41.7% ensemble number is a parser artifact, not a quality signal.
- Value probe (small N): on hard problems where cheap models are individually
  unreliable, the ensemble went 9/9, **beating the best cheap single (+11 pts)**,
  but **tied a single strong model (Sonnet) at ~6× the cost.** Mechanism confirmed:
  cross-examination corrects individual errors, not just majority voting.

## Goals for tomorrow

1. Get the **honest headline** debate-vs-model number on `harder.jsonl` now that
   stalls are fixed.
2. Turn the value finding from a clean signal (N=9) into a **firm claim** (N≥30).
3. Shave the **6× cost gap** if possible without losing accuracy.

## Plan

### 1. Clean full re-run of `harder.jsonl` (the do-over)
- Re-run all 72 with the stall fix in place:
  `ensemble-eval --dataset evals/harder.jsonl --models gpt4o,claude,deepseek
   --baseline sonnet --delay 2 --stall-timeout 120 --log evals/harder-run-v2.jsonl`
- Expect the non-consensus count to collapse from 41 → near 0. Report the real
  ensemble accuracy and **replace the pre-fix table** in the README "harder run"
  section (keep a one-line note that the first run was parser-bugged).
- Caveat to keep stating: these Qs are easy enough that cheap singles already
  score ~90%, so don't expect the ensemble to separate from the field here.

### 2. Build `evals/value.jsonl` — the "cheap-models-unreliable" set (N≥30)
- This is the regime where debate actually earns its cost. Use the **two-stage
  method** that worked today:
  - Stage A (cheap): probe gpt4o/claude/deepseek **solo, 3× each** on a large pool
    of hard, objective, auto-gradeable problems; **self-verify every answer in
    Python** at build time (reuse the `build_harder.py` assert pattern).
  - Keep questions where the cheap models are **individually unreliable** — target
    "≤1 of 3 models correct on average," especially the *lone-correct* cases (one
    model right, two wrong) that stress whether debate follows truth or the
    majority. Aim for ≥30 such questions across math/logic/counting/combinatorics.
- Save the verified set + a `build_value.py` generator. Today's seeds to include:
  factorial sum (5039), squares-or-cubes ≤1000 (38), cryptarithm `AB×6=BBB` (11),
  strictly-increasing 4-digit count (126).

### 3. Run the value comparison with rigor
- 3 cheap singles + ensemble + Sonnet baseline, **2–3× per question** (temp 0.7),
  report **mean ± spread** and **cost-per-correct**, per the original rigor goal.
- Headline metrics: ensemble vs best-cheap-single (expect a real win) and ensemble
  vs Sonnet (expect a tie at higher cost). Log everything for audit.

### 4. Cost reduction experiment (does debate need all four phases?)
- The ensemble is ~6× a strong model largely because each question is ~10–15 calls
  (3 models × propose/review/rebuttal/vote, growing prompts). Test a **leaner
  debate**: propose → vote only (drop review+rebuttal) and propose → review → vote.
- Measure the accuracy/cost trade-off on the value set. If a 2-phase debate keeps
  most of the accuracy at half the cost, that materially improves the value case.

### 5. Reliability hardening (small, finish the debugging)
- Investigate the **1 remaining** stall (a genuinely missing vote file — likely an
  agent that errored without hitting `MAX_PHASE_FAILURES`). Decide whether the
  coordinator should treat "phase file absent past a grace period" as a drop.
- Add a defensive log/metric: a debate that stalls in voting **with all phase
  files present** should now be impossible — assert/log loudly if it recurs.

### 6. Housekeeping
- Decide whether to commit the day-1 work (branch + PR): grading fix, baseline,
  logging, **the vote-parsing stall fix**, dataset, README Evaluation section.
- Confirm `evals/.env` and `evals/*-run.jsonl` stay git-ignored (they do).

## Success criteria

- A clean `harder.jsonl` number with ~0 stalls (verdict: ensemble ≈ cheap singles
  on easy objective Qs — expected, low-headroom).
- On `value.jsonl` (N≥30): a **statistically meaningful** ensemble win over the
  best cheap single, and a clear read on the ensemble-vs-Sonnet accuracy/cost
  frontier.
- A decision on the leaner-debate cost trade-off.
- If the data says "a single strong model is the better buy," that's a valid and
  important finding to state plainly.

## Checklist

- [ ] Re-run `harder.jsonl` post-fix; update README table + note.
- [ ] Build + verify `evals/value.jsonl` (≥30) via two-stage probe + `build_value.py`.
- [ ] Run value comparison 2–3× with mean±spread and cost/correct.
- [ ] Leaner-debate (2-phase / 3-phase) cost-vs-accuracy experiment.
- [ ] Investigate the lone remaining missing-vote stall; add the recurrence guard.
- [ ] Commit day-1 work on a branch / open PR.
- [ ] Use fresh, rotated API keys (today's are in `evals/.env`).
