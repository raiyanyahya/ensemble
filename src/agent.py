from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .models import PROVIDERS, default_model_id, get_api_key
from .state import (
    PHASE_FILE_SUFFIX,
    DebateState,
    Phase,
    assign_aliases,
    atomic_write_text,
    load_state,
)

log = logging.getLogger("ensemble.agent")

# After this many consecutive failures in a single phase, the agent stops
# trying. The coordinator's stall timeout then ends the debate gracefully
# rather than every loop spinning forever.
MAX_PHASE_FAILURES = 4


SYSTEM_PROMPT = """You are {alias}, one of several independent participants in a structured,
multi-round discussion working toward a well-reasoned shared conclusion.
The other participants' identities are hidden from you. Judge every contribution purely on
the strength of its reasoning — never on assumptions about who or what produced it. You are
all equals; decisions are made by majority vote.

COMMUNICATION PROTOCOL:
You communicate ONLY by reading and writing files in a shared folder.
You cannot speak to anyone directly. Read their writing to understand their position.

THE PROCESS:
1. PROPOSING: independently write your analysis and proposed answer.
2. REVIEWING: review the other participants' proposals — critique, find gaps, acknowledge
   strong points. You may pose a direct question to a specific participant
   (e.g. "To Participant B: how does your approach handle X?").
3. REBUTTAL: respond to critiques of your own proposal and answer any questions directed at
   you. Defend, concede, or refine — this is your chance to reply before any vote.
4. VOTING: vote on the next step:
   - FINALIZE: <participant> — endorse the single best proposal by its label
   - REVISE: <focus> — another round is needed, with a stated focus
   - SPLIT: <reason> — there is a fundamental disagreement
   You may also rank all participants in a `## Ranking` line (best→worst).
5. SYNTHESIS (only the endorsed author): write one merged answer that integrates the
   strongest points across all proposals and explicitly preserves any minority view.
   End with a line `Final answer: <answer>`, stating it in exactly the format the
   question requested (e.g. only a number or a single word) so the conclusion is
   unambiguous even after the explanation.
6. CONFIRM: judge whether that synthesis faithfully captures the group's conclusion —
   reply APPROVE or REJECT.

RULES:
- Be intellectually honest. If another participant has the better argument, say so.
- Do not agree just to reach consensus. Justify your position.
- When voting FINALIZE, you MUST name the proposal you endorse by its label (e.g. "FINALIZE:
  Participant B"); you may endorse your own only if it is genuinely the strongest.
- Up to {max_rounds} round(s). If no proposal earns a majority endorsement, the discussion
  continues or, at the round limit, ends without consensus.

IMPORTANT — YOUR OUTPUT FORMAT (follow exactly):

## Proposal
[Your independent analysis and proposed answer.]
(Only in the PROPOSING phase.)

## Reviews
[Review each other participant's proposal; pose direct questions if useful.]
(Only in the REVIEWING phase.)

## Rebuttal
[Respond to critiques of your proposal and answer questions directed at you.]
(Only in the REBUTTAL phase.)

## Vote
[Your vote MUST be the first line of this section, exactly one of:
 FINALIZE: <participant label>
 REVISE: <focus>
 SPLIT: <reason>]
(Only in the VOTING phase.)

## Reasoning
[Why you chose this vote, and — if FINALIZE — why that proposal is strongest.]
(Only in the VOTING phase.)

## Ranking
[Optional, VOTING only: rank ALL participant labels best-to-worst, e.g. "B > C > A".]

## Synthesis
[SYNTHESIS phase only: the single best merged answer, integrating the strongest points
 across proposals and explicitly preserving any minority view. End with a line
 `Final answer: <answer>` in exactly the format the question requested (e.g. only a
 number or single word), after any explanation.]

## Confirm
[CONFIRM phase only: the first line must be exactly APPROVE or REJECT.]"""


# Past rounds are condensed by truncation to bound prompt growth; the *current*
# round is shown to peers in full so they always judge the complete contribution.
PAST_PROPOSAL_CHARS = 6000
PAST_REVIEW_CHARS = 3000


def _agent_participates(model_name: str, state: DebateState) -> bool:
    """Whether this model should contribute to the current phase.

    SYNTHESIS is authored by the endorsed (winning) model alone; every other phase
    is open to all active participants.
    """
    rs = state.round_state
    if rs.phase == Phase.SYNTHESIS:
        return model_name == rs.consensus_winner
    return True


def _alias(state: DebateState, name: str) -> str:
    """The identity-free label for a model, falling back to a derived one."""
    return (
        state.participant_aliases.get(name)
        or assign_aliases(state.active_models).get(name, name)
    )


def build_agent_prompt(model_name: str, state: DebateState) -> tuple[str, str]:
    system = SYSTEM_PROMPT.format(alias=_alias(state, model_name), max_rounds=state.max_rounds)

    rs = state.round_state

    context_parts = [
        f"DISCUSSION: {state.debate_id}",
        f"PROMPT: {state.prompt}",
        f"ROUND: {state.current_round}/{state.max_rounds}",
        f"PHASE: {rs.phase.value}",
        f"FOCUS (if REVISE): {rs.focus or 'N/A'}",
    ]

    role = state.roles.get(model_name)
    if role:
        context_parts.append(
            f"\nYOUR ASSIGNED STANCE: {role}\n"
            "Argue genuinely from this stance, but stay intellectually honest — "
            "concede when a peer is right."
        )

    if state.sources:
        src_lines = ["\nGROUNDING SOURCES (cite as [n] for any factual claim):"]
        for i, s in enumerate(state.sources, 1):
            src_lines.append(f"[{i}] {s.title} — {s.url}\n    {s.snippet}")
        context_parts.extend(src_lines)

    def _votes_by_alias(votes: dict) -> dict:
        return {_alias(state, k): (v.value if hasattr(v, "value") else v) for k, v in votes.items()}

    for past in state.rounds[:-1]:
        context_parts.append(f"\n--- ROUND {past.round_num} RESULTS ---")
        for name, output in past.model_outputs.items():
            label = _alias(state, name)
            context_parts.append(
                f"\n### {label}'s Proposal:\n{output.proposal[:PAST_PROPOSAL_CHARS]}"
            )
            if output.reviews:
                context_parts.append(
                    f"\n### {label}'s Reviews:\n{output.reviews[:PAST_REVIEW_CHARS]}"
                )
        if past.votes:
            context_parts.append(f"\n### Votes: {_votes_by_alias(past.votes)}")
        if past.consensus_action:
            context_parts.append(f"\n### Outcome: {past.consensus_action}")

    current_files = []
    for name, output in rs.model_outputs.items():
        if name == model_name:
            continue
        label = _alias(state, name)
        # Current round shown in full — peers must see the complete contribution.
        if output.proposal:
            current_files.append(f"\n### {label}'s Proposal:\n{output.proposal}")
        if output.reviews:
            current_files.append(f"\n### {label}'s Reviews:\n{output.reviews}")
        if output.rebuttal:
            current_files.append(f"\n### {label}'s Rebuttal:\n{output.rebuttal}")

    if current_files:
        context_parts.append("\n\nCURRENT ROUND — OTHER PARTICIPANTS' CONTRIBUTIONS:")
        context_parts.extend(current_files)

    if rs.votes:
        context_parts.append(f"\n\nCURRENT VOTES: {_votes_by_alias(rs.votes)}")

    # During CONFIRM, every participant needs the synthesis text to judge it.
    if rs.phase == Phase.CONFIRM:
        syn = rs.model_outputs.get(rs.consensus_winner)
        if syn and syn.synthesis:
            context_parts.append(f"\n\nPROPOSED SYNTHESIS TO CONFIRM:\n{syn.synthesis}")

    cite = ""
    if state.sources:
        cite = " Ground factual claims in the sources above and cite them as [n]."
    context_parts.append(
        f"\n\nYOUR TASK: You are in the {rs.phase.value.upper()} phase. "
        f"Write your output following the format specified in the system prompt.{cite}"
    )

    return system, "\n".join(context_parts)


async def agent_loop(
    model_name: str,
    debate_dir: Path,
    poll_interval: float = 2.0,
) -> None:
    if not get_api_key(model_name):
        log.debug("%s: no API key, sitting out", model_name)
        return

    call_fn = PROVIDERS[model_name].call
    state_path = debate_dir / "state.json"

    contributed_phases: set[str] = set()
    failures: dict[str, int] = {}

    while True:
        if not state_path.exists():
            await asyncio.sleep(poll_interval)
            continue

        try:
            state = load_state(state_path)
        except Exception as e:  # torn read or transient corruption: retry
            log.debug("%s: state read failed (%s), retrying", model_name, e)
            await asyncio.sleep(poll_interval)
            continue

        if state.is_finished:
            break

        # The coordinator may have dropped us after we gave up on a phase; if so,
        # stop looping rather than keep burning calls.
        if model_name not in state.active_models:
            log.info("%s: dropped from the debate, stopping", model_name)
            break

        rs = state.round_state
        if rs.phase not in PHASE_FILE_SUFFIX:
            await asyncio.sleep(poll_interval)
            continue

        # The endorsed author writes the synthesis alone; others wait it out.
        if not _agent_participates(model_name, state):
            await asyncio.sleep(poll_interval)
            continue

        phase_key = f"r{state.current_round}-{rs.phase.value}"
        if phase_key in contributed_phases:
            await asyncio.sleep(poll_interval)
            continue

        existing = rs.model_outputs.get(model_name)
        if existing is not None and existing.has_phase(rs.phase):
            contributed_phases.add(phase_key)
            await asyncio.sleep(poll_interval)
            continue

        log.info("%s: %s — round %d", model_name, rs.phase.value, state.current_round)

        system, user = build_agent_prompt(model_name, state)
        model_id = state.model_ids.get(model_name) or default_model_id(model_name)
        try:
            result = await call_fn(system, user, model_id)
        except Exception as e:
            failures[phase_key] = failures.get(phase_key, 0) + 1
            log.warning(
                "%s: API error (%d/%d): %s",
                model_name, failures[phase_key], MAX_PHASE_FAILURES, e,
            )
            if failures[phase_key] >= MAX_PHASE_FAILURES:
                log.error("%s: giving up on %s after %d failures; signaling failure",
                          model_name, phase_key, MAX_PHASE_FAILURES)
                # Leave a sentinel so the coordinator can drop us and let the
                # debate proceed with the remaining models instead of stalling.
                suffix = PHASE_FILE_SUFFIX[rs.phase]
                atomic_write_text(
                    debate_dir / f"round-{rs.round_num:03d}" / f"{model_name}.{suffix}.failed",
                    str(e)[:500],
                )
                return
            await asyncio.sleep(poll_interval)
            continue

        # Providers return a ModelResult; tolerate a plain str (simple/fake calls).
        text = result if isinstance(result, str) else result.text
        usage = None if isinstance(result, str) else result.usage

        round_dir = debate_dir / f"round-{rs.round_num:03d}"
        suffix = PHASE_FILE_SUFFIX[rs.phase]
        atomic_write_text(round_dir / f"{model_name}.{suffix}.md", text)
        if usage is not None:
            # Usage sidecar; the coordinator (sole state.json writer) tallies it.
            atomic_write_text(
                round_dir / f"{model_name}.{suffix}.usage.json",
                json.dumps({
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "cache_creation_tokens": usage.cache_creation_tokens,
                    "model_id": model_id,
                }),
            )
        contributed_phases.add(phase_key)

        await asyncio.sleep(poll_interval)
