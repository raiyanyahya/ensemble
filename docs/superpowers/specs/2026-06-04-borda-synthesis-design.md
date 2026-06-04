# Design: Additive Borda Ranking + Synthesis-as-Candidate

**Date:** 2026-06-04
**Status:** Approved
**Inspiration:** karpathy/llm-council (anonymous ranking + chairman synthesis), adapted
to ensemble's multi-round, consensus-by-vote, filesystem-artifact model.

## Goal

Borrow two ideas from llm-council without breaking ensemble's documented semantics:

1. **Ordinal (Borda) ranking** — a richer per-participant signal than the single
   FINALIZE/REVISE/SPLIT directive.
2. **Synthesis-as-candidate** — after consensus, produce a *merged* answer that
   integrates the strongest points (including minority views) instead of shipping the
   winning proposal verbatim. Crucially **not** a single "chairman arbiter": the merge
   is a candidate that the participants confirm by vote.

Both are **purely additive**: the existing FINALIZE/REVISE/SPLIT consensus mechanism is
unchanged, and the worst-case output equals today's behavior.

## Non-goals

- Borda does **not** replace or alter the consensus decision on the finalize path.
- No new web UI; artifacts stay on the filesystem.
- No OpenRouter backend (deferred to a separate effort).

## A. Borda ranking

### Source — piggyback on the VOTING phase
- The VOTING phase prompt gains an instruction to optionally append a ranking.
- The vote file gains an optional `## Ranking` section, e.g. `B > C > A` (best→worst, by
  participant label).
- Parsing is **tolerant**, mirroring the existing headerless-vote recovery
  (`coordinator.py:_headerless_vote_section` / `detect_vote`): a missing, partial, or
  garbled ranking simply contributes no ballot — it never stalls the debate. A ranking
  that lists a subset of labels is accepted; unlisted labels score 0 for that ballot.

### Tally
- Standard Borda: with `k` ranked proposals on a ballot, the proposal at 0-based rank `i`
  scores `k - 1 - i`.
- Aggregate across all valid ballots into `borda_scores: dict[label, int]`, stored on the
  round's state.

### What Borda is allowed to affect (additive only)
1. **Recorded signal** — always written to `state.json` and surfaced in `final.md` as a
   ranking block; exposed to `ensemble-eval`.
2. **Narrow tiebreaker** — used *only* where today's logic is already arbitrary: a
   **plurality tie on a deadlock** (two+ proposals with equal top endorsements). Borda
   breaks it deterministically instead of the current first-max-wins. If Borda also ties,
   fall back to the current deterministic ordering (stable).

Borda does **not** touch the finalize/consensus path, which already yields a unique
majority winner. FINALIZE semantics and the documented open-weights example are untouched.

## B. Synthesis-as-candidate

Runs **only** when `determine_consensus → "finalize"`. Two new terminal phases execute
after VOTING and before `write_final_answer`:

### SYNTHESIS phase
- The **endorsed author** (the winning model) receives all proposals + critiques (peers
  still anonymized as Participant X) and the fact that its proposal won.
- It writes `round-NNN/<winner>.synthesis.md`: a merged final answer integrating the
  strongest points across proposals, with **minority views explicitly preserved**.

### CONFIRM phase
- Each active participant reads the synthesis (author anonymized) and writes
  `round-NNN/<model>.confirm.md` containing `APPROVE` or `REJECT` (tolerant parse,
  first-directive-wins, consistent with `detect_vote`).

### Gate
- `APPROVE >= state.majority()` (existing threshold; the author's own APPROVE counts) →
  `final.md` uses the synthesis.
- Otherwise, or if the synth author errors, or if confirm votes stall → **fall back to the
  verbatim winning proposal** (today's exact `render_final_answer` output).

So the worst case is precisely current behavior.

### Deadlock
Unchanged. No synthesis. Plurality proposal written as best-effort exactly as today (with
the Borda tiebreak from §A applied only to a plurality *tie*).

## C. Cross-cutting

### Artifacts
- `round-NNN/` gains `<winner>.synthesis.md` and `<model>.confirm.md`.
- `final.md` states whether synthesis was used or fell back, and shows the Borda ranking.
- Consistent with the existing one-file-per-phase convention; fully auditable.

### State (`state.py`)
New fields on `RoundState`, all written atomically and resumable:
- `rankings: dict[str, list[str]]` — raw ballots by model.
- `borda_scores: dict[str, int]`.
- `synthesis_used: bool`.
- `synthesis_author: str | None`.
- `confirm_tally: dict[str, int]` — APPROVE/REJECT counts.

An interrupted debate resumes mid-synthesis (e.g. SYNTHESIS written, CONFIRM pending).

### Phase enum
Add `SYNTHESIS` and `CONFIRM`. They are entered **only** on the finalize path; all other
terminal paths (deadlock, split, budget/round fuse) skip them.

### Cost
- Borda: **0** extra API calls.
- Synthesis: 1 synthesis call + N short confirm calls, **only on successful consensus** —
  never on the stall/deadlock paths the eval is sensitive to.

### Testing (`tests/test_flow.py` + units)
- Extend the fake providers to emit `## Ranking`, a synthesis, and confirm votes.
- End-to-end: synthesis-used-on-approve; fallback-on-reject; fallback-on-author-failure.
- Units: Borda tally math; deadlock plurality-tie tiebreaker; tolerant `## Ranking`
  parsing (missing/partial/garbled); confirm-vote parsing.

## Touch points (for the implementation plan)
- `src/state.py` — Phase enum, RoundState fields.
- `src/coordinator.py` — ranking parse + Borda tally, tiebreak in `_endorsement_tally`/
  deadlock path, synthesis/confirm orchestration in `coordinator_loop`, `final.md`
  rendering.
- `src/agent.py` — VOTING prompt ranking instruction; SYNTHESIS and CONFIRM phase prompts.
- `tests/test_flow.py` — extended fakes + new assertions; new unit tests.
</content>
</invoke>
