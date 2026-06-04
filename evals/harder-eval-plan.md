# Harder evaluation — plan

_Status: **executed 2026-06-04.** Dataset (`harder.jsonl`, N=72), strong-model
baseline (Sonnet 4.6), per-question logging, reliability knobs, and the grading
fix are all built and shipped; the full single pass ran on fresh keys. Results +
honest writeup are in the README "Evaluation" section. **Headline finding: the
run is gated by a convergence-reliability bug — 40/72 quick debates stalled in
voting and timed out. On the 31 that converged the ensemble hit 93.5% (≈ Sonnet,
> the cheap singles), but that subset is selection-biased.**

**Stall root cause found + fixed (2026-06-04):** 45/138 vote files held a valid
directive with no `## Vote` header; the parser dropped them and the coordinator
waited out the timeout. `parse_output` now recovers unwrapped directives — all
137 recorded vote files re-parse, and a live previously-stalling question reached
consensus. **Next: clean full re-run, then make the debate-vs-model claim.**
Owner: raiyan._

## Why

We now have two live eval runs (via `ensemble-eval`):

| Dataset | gpt4o | claude | deepseek | **Ensemble** | Ensemble cost |
|---|---|---|---|---|---|
| `QUESTIONS` (easy, built-in) | 93% | 100% | 100% | **100%** | $0.21 (91×) |
| `evals/hard.jsonl` (15 traps) | 60% | 93% | 93% | **100%** | $0.17 (134×) |

The hard set produced the first real signal: the ensemble matched-or-beat the best
single model and beat the weakest by **+40 points**. But the result is **not yet
conclusive**, for three reasons this plan must fix.

## Problems with the current eval

1. **Sample too small.** 15 questions → "ensemble beat best model by 6.7 pts" is a
   *single question*. We can't distinguish skill from noise.
2. **No strong-model baseline.** The real question isn't "3 cheap models vs 1 cheap
   model" — it's **"3 cheap models debating vs 1 strong model answering once."** If a
   single GPT-4o/Sonnet-class call matches the ensemble at a fraction of the cost,
   the ensemble's value case weakens. We haven't tested this.
3. **Reliability.** 5 of 15 ensemble debates hit the 60s stall timeout (likely rate
   limiting from bursting ~225 calls). They degraded to best-effort answers that
   happened to be correct — but a third of the "wins" weren't clean consensus.
4. **Grading is substring-based** on verbose proposals, which can mildly favor the
   (longer) ensemble output.

## Goals for the harder eval

- Get a **statistically meaningful** verdict on whether debate > single model.
- Include a **strong single model** as the baseline that actually matters.
- Make the run **reliable** (no stall-driven deadlocks distorting results).
- Tighten **grading** so the comparison is fair.

## Plan

### 1. Dataset (`evals/harder.jsonl`, target N ≥ 60)
- Pull from genuinely hard, **objective, auto-gradeable** sources across categories so
  no single skill dominates:
  - multi-step math word problems (GSM8K-hard style),
  - logic / deduction puzzles,
  - counting & string manipulation (LLM-weak),
  - factual edge cases / units / dates,
  - "trap" questions where the intuitive answer is wrong.
- Each item: `{"question": ..., "accept": [variants...]}`. Phrase every question to
  demand a **bare final answer** ("Reply with only the number/word").
- Hand-verify every answer (a wrong key silently corrupts the whole eval).

### 2. Add a strong-model baseline
- Use the existing model-id override to add a 4th "condition": a single **strong**
  model (e.g. `gpt-4o` full or `claude-sonnet`-class), one call per question.
- Report it alongside the three cheap singles and the ensemble. **This is the headline
  comparison.**
- (Stretch) also run an *ensemble that includes the strong model* to see if mixing
  tiers beats either — and whether it reduces correlated errors.

### 3. Reliability fixes (do before the run)
- Investigate the 5 stalls: almost certainly transient rate limits under burst load.
- Options to add to the harness: a small inter-question delay / concurrency cap,
  higher retry budget, and/or longer stall timeout *with* exponential backoff.
- Log per-question outcomes to a file (currently only booleans are kept in memory) so
  we can see *which* questions each condition missed and audit the deadlocks.

### 4. Grading improvements
- Extract the model's **final answer token** (last number/word, or a constrained
  answer line) rather than whole-text substring search, to remove the verbosity bias.
- Keep deterministic auto-grading for objective questions; consider an LLM-judge only
  if we later add open-ended questions.

### 5. Rigor
- Run each condition **2–3×** (temperature 0.7) and report mean ± spread, since today
  we only spot-checked reproducibility on one question.
- Report **cost per correct answer** and the cost multiple vs. the best single model,
  not just accuracy.

## Success criteria

We can make a defensible claim if, on N ≥ 60 hard questions:
- the ensemble's accuracy is **meaningfully** above the best *single cheap* model, and
- we know how it compares to a single **strong** model on the accuracy/cost frontier.

If the ensemble doesn't beat a single strong model, that's a valid (and important)
finding too — it would mean "buy a better model" beats "debate cheaper models."

## Checklist for tomorrow

- [ ] Build `evals/harder.jsonl` (≥60 verified questions, categorized).
- [ ] Add strong-model baseline condition to `src/eval.py` (`--baseline gpt-4o` style).
- [ ] Add per-question result logging + reliability fix (delay/concurrency cap).
- [ ] Improve answer extraction in `grade()`.
- [ ] Run; capture the table + cost/correct; note any deadlocks.
- [ ] Write up results (and add an honest "Evaluation" section to the README).
- [ ] Use fresh, rotated API keys.
