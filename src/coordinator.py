from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path

from .models import Usage, default_model_id, provider_name, usage_cost
from .state import (
    MIN_MODELS,
    PHASE_FILE_SUFFIX,
    DebateState,
    ModelOutput,
    ModelUsage,
    Phase,
    RoundState,
    Vote,
    assign_aliases,
    atomic_write_text,
    load_state,
    now_iso,
    save_state,
)

log = logging.getLogger("ensemble.coordinator")

DEBATES_DIR = Path.home() / ".ensemble" / "debates"

# If no phase completes within this many seconds, the debate is declared
# deadlocked instead of hanging forever (e.g. a provider is down).
DEFAULT_STALL_TIMEOUT = 300.0


def create_debate(prompt: str) -> DebateState:
    stamp = now_iso()[:19].replace(":", "").replace("-", "").replace("T", "-")
    debate_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
    return DebateState(
        debate_id=debate_id,
        prompt=prompt,
        created_at=now_iso(),
        rounds=[RoundState(round_num=1, phase=Phase.PROPOSING)],
        current_round=1,
        current_phase=Phase.PROPOSING,
    )


def setup_debate_dir(state: DebateState) -> Path:
    debate_dir = DEBATES_DIR / state.debate_id
    debate_dir.mkdir(parents=True, exist_ok=True)
    debate_dir.chmod(0o700)
    (debate_dir / "prompt.md").write_text(f"# Debate Prompt\n\n{state.prompt}\n")
    save_state(state, debate_dir / "state.json")
    return debate_dir


# --- Parsing -----------------------------------------------------------------


# A section runs until the next *known* section header (or end of text). Using
# only known headers as boundaries — not a bare "##" — means a model's own
# Markdown sub-headings inside a section (## Architecture, ## Tradeoffs, …) are
# preserved rather than truncated at the first one.
_SECTION_END = (
    r"(?=\n##\s*(?:Proposal|Reviews?|Rebuttal|Vote|Reasoning|Ranking|Synthesis|Confirm)\b|\Z)"
)


def _section(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None


# A bare vote directive at the start of a line (after optional markdown noise
# like "> ", "*", "`"). Models routinely emit this WITHOUT the "## Vote" header,
# because the format tells them the vote MUST be the first line of the section.
_VOTE_DIRECTIVE = re.compile(r"(?im)^[\s>*_`-]*(?:FINALIZE|REVISE|SPLIT)\b")


def _headerless_vote_section(text: str) -> str | None:
    """Recover a vote written without a ``## Vote`` header.

    Returns the slice from the first directive line up to the next known section
    header (e.g. ``## Reasoning``), or ``None`` if no directive is present. This
    is what stops the debate from stalling: an unwrapped-but-valid vote is parsed
    instead of being silently dropped (the agent's call succeeded, so it leaves
    no failure sentinel, and the coordinator would otherwise wait forever).
    """
    m = _VOTE_DIRECTIVE.search(text)
    if not m:
        return None
    tail = text[m.start():]
    end = re.search(
        r"\n##\s*(?:Proposal|Reviews?|Rebuttal|Vote|Reasoning|Ranking|Synthesis|Confirm)\b",
        tail,
    )
    return (tail[: end.start()] if end else tail).strip() or None


# A bare confirm directive (APPROVE / REJECT), with the same tolerance as votes:
# models often drop the ``## Confirm`` header and just write the verdict.
_CONFIRM_DIRECTIVE = re.compile(r"(?im)^[\s>*_`-]*(APPROVE|REJECT)\b")


def detect_confirm(text: str) -> str:
    """Recover an APPROVE/REJECT verdict from confirm text (header optional)."""
    m = _CONFIRM_DIRECTIVE.search(text or "")
    return m.group(1).upper() if m else ""


def parse_ranking(section: str) -> list[str]:
    """Parse a ``B > C > A`` ranking line into ordered label tokens."""
    line = next((ln.strip() for ln in section.splitlines() if ln.strip()), "")
    return [t.strip() for t in re.split(r"[>›]", line) if t.strip()]


def detect_vote(section: str) -> Vote | None:
    """Pick the vote from a Vote section.

    Prefer the first non-empty line (the format instructs models to put the
    bare vote there); fall back to the earliest keyword anywhere in the
    section. Earliest-position wins so "I won't FINALIZE, I vote REVISE"
    resolves to whichever the model actually leads with.
    """
    lines = [ln.strip() for ln in section.splitlines() if ln.strip()]
    for text in ([lines[0]] if lines else []) + [section]:
        upper = text.upper()
        positions = {v: upper.find(v.value.upper()) for v in Vote}
        positions = {v: i for v, i in positions.items() if i != -1}
        if positions:
            return min(positions, key=lambda v: positions[v])
    return None


def parse_output(text: str) -> ModelOutput:
    output = ModelOutput()

    prop = _section(text, rf"##\s*Proposal\s*\n(.*?){_SECTION_END}")
    if prop:
        output.proposal = prop

    rev = _section(text, rf"##\s*Reviews?\s*\n(.*?){_SECTION_END}")
    if rev:
        output.reviews = rev

    reb = _section(text, rf"##\s*Rebuttal\s*\n(.*?){_SECTION_END}")
    if reb:
        output.rebuttal = reb

    vote_sec = _section(text, rf"##\s*Vote\s*\n(.*?){_SECTION_END}")
    vote = detect_vote(vote_sec) if vote_sec else None
    if vote is None:
        # Header missing, empty, or garbled — recover an unwrapped directive so a
        # present vote isn't lost (the dominant cause of voting-phase stalls).
        recovered = _headerless_vote_section(text)
        if recovered:
            vote_sec, vote = recovered, detect_vote(recovered)
    if vote is not None:
        output.vote_reasoning = vote_sec or ""
        output.vote = vote

    rank_sec = _section(text, rf"##\s*Ranking\s*\n(.*?){_SECTION_END}")
    if rank_sec:
        output.ranking = parse_ranking(rank_sec)

    syn = _section(text, rf"##\s*Synthesis\s*\n(.*?){_SECTION_END}")
    if syn:
        output.synthesis = syn

    confirm_sec = _section(text, rf"##\s*Confirm\s*\n(.*?){_SECTION_END}")
    # Prefer a verdict inside the Confirm section; fall back to a bare directive
    # anywhere (header omitted), mirroring the tolerant vote recovery.
    output.confirm = detect_confirm(confirm_sec) if confirm_sec else detect_confirm(text)

    return output


def _phase_file(debate_dir: Path, state: DebateState, name: str) -> Path:
    rs = state.round_state
    suffix = PHASE_FILE_SUFFIX[rs.phase]
    return debate_dir / f"round-{rs.round_num:03d}" / f"{name}.{suffix}.md"


def read_phase_outputs(debate_dir: Path, state: DebateState) -> dict[str, ModelOutput]:
    outputs: dict[str, ModelOutput] = {}
    for name in state.active_models:
        f = _phase_file(debate_dir, state, name)
        if f.exists():
            try:
                # Only harvest the field this phase owns, so a section emitted
                # out of turn (e.g. a vote leaked into a review) is ignored.
                outputs[name] = parse_output(f.read_text()).for_phase(state.round_state.phase)
            except OSError:
                continue
    return outputs


def accumulate_usage(debate_dir: Path, state: DebateState) -> None:
    """Fold this phase's per-model usage sidecars into ``state.usage`` (once)."""
    rs = state.round_state
    suffix = PHASE_FILE_SUFFIX[rs.phase]
    for name in state.active_models:
        f = debate_dir / f"round-{rs.round_num:03d}" / f"{name}.{suffix}.usage.json"
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):  # a crafted/corrupt sidecar must not crash the loop
            continue
        usage = Usage(
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cached_tokens=d.get("cached_tokens", 0),
            cache_creation_tokens=d.get("cache_creation_tokens", 0),
        )
        model_id = d.get("model_id", default_model_id(name))
        cost = usage_cost(name, model_id, usage)
        state.usage.setdefault(name, ModelUsage()).add(
            usage.input_tokens, usage.output_tokens, usage.cached_tokens, cost
        )


def _failed_sentinel(debate_dir: Path, state: DebateState, name: str) -> Path:
    rs = state.round_state
    suffix = PHASE_FILE_SUFFIX[rs.phase]
    return debate_dir / f"round-{rs.round_num:03d}" / f"{name}.{suffix}.failed"


def failed_models(debate_dir: Path, state: DebateState) -> list[str]:
    """Active models whose agent left a ``.failed`` sentinel for this phase."""
    return [n for n in state.active_models if _failed_sentinel(debate_dir, state, n).exists()]


def failure_reason(debate_dir: Path, state: DebateState, name: str) -> str:
    try:
        return _failed_sentinel(debate_dir, state, name).read_text().strip()[:200] or "unavailable"
    except OSError:
        return "unavailable"


def salvage_phase_outputs(debate_dir: Path, state: DebateState) -> None:
    """Best-effort merge of on-disk phase contributions into state, so a deadlock
    still surfaces work that was written but never committed by a phase advance."""
    rs = state.round_state
    for name, output in read_phase_outputs(debate_dir, state).items():
        rs.model_outputs.setdefault(name, ModelOutput()).merge_from(output)


def _phase_participants(state: DebateState) -> list[str]:
    """Models expected to contribute to the current phase.

    SYNTHESIS is authored by the endorsed winner alone; all other phases expect
    every active model.
    """
    rs = state.round_state
    if rs.phase == Phase.SYNTHESIS and rs.consensus_winner:
        return [rs.consensus_winner]
    return state.active_models


def all_phase_complete(debate_dir: Path, state: DebateState) -> bool:
    rs = state.round_state
    for name in _phase_participants(state):
        f = _phase_file(debate_dir, state, name)
        if not f.exists():
            return False
        try:
            output = parse_output(f.read_text())
        except OSError:
            return False
        if not output.has_phase(rs.phase):
            return False
    return True


# --- Consensus ---------------------------------------------------------------


def tally_votes(state: DebateState) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in state.round_state.votes.values():
        key = v.value if isinstance(v, Vote) else str(v)
        counts[key] = counts.get(key, 0) + 1
    return counts


def resolve_endorsement(reasoning: str, state: DebateState, voter: str) -> str:
    """Which proposal a FINALIZE vote endorses, by participant label.

    Reads the target after ``FINALIZE:`` and matches it to an active model — first
    by full label ("Participant B"), then by a bare letter token ("B"). Falls back
    to the voter's own proposal if nothing resolvable is named, so a vague finalize
    counts as "I prefer my own answer" (and thus won't manufacture false agreement).
    """
    aliases = state.participant_aliases or assign_aliases(state.active_models)
    m = re.search(r"FINALIZE\b[:\s]*(.+)", reasoning or "", re.IGNORECASE)
    target = (m.group(1).splitlines()[0] if m else "").strip().upper()
    if target:
        for name in state.active_models:  # prefer a full-label match
            if aliases.get(name, name).upper() in target:
                return name
        for name in state.active_models:  # then a bare letter ("B")
            letter = aliases.get(name, name).split()[-1].upper()
            if re.search(rf"\b{re.escape(letter)}\b", target):
                return name
    return voter


def _resolve_label(token: str, state: DebateState) -> str | None:
    """Match a ranking token ("B" / "Participant B") to an active model key."""
    aliases = state.participant_aliases or assign_aliases(state.active_models)
    up = token.upper()
    for name in state.active_models:  # prefer a full-label match
        if aliases.get(name, name).upper() in up:
            return name
    for name in state.active_models:  # then a bare letter ("B")
        letter = aliases.get(name, name).split()[-1].upper()
        if re.search(rf"\b{re.escape(letter)}\b", up):
            return name
    return None


def _resolve_ranking(tokens: list[str], state: DebateState) -> list[str]:
    """Resolve a ballot of label tokens to a de-duplicated list of model keys."""
    out: list[str] = []
    for t in tokens:
        m = _resolve_label(t, state)
        if m and m not in out:
            out.append(m)
    return out


def compute_borda(state: DebateState) -> dict[str, int]:
    """Aggregate Borda points across every ranking ballot in the current round.

    On a ballot of ``k`` ranked proposals, rank ``i`` (0-based) scores ``k-1-i``.
    Partial ballots are fine; unranked proposals simply earn nothing from them.
    """
    rs = state.round_state
    scores = {m: 0 for m in state.active_models}
    for out in rs.model_outputs.values():
        ballot = _resolve_ranking(out.ranking, state)
        k = len(ballot)
        for i, m in enumerate(ballot):
            scores[m] += (k - 1 - i)
    return scores


def _endorsement_tally(state: DebateState) -> tuple[str | None, int]:
    """The most-endorsed proposal and its endorsement count (or (None, 0)).

    A plurality tie (e.g. 1–1 on a deadlock) is broken by Borda score — the only
    case where the ranking signal affects an outcome, and one that was previously
    decided arbitrarily by dict order. A real majority is always unique, so the
    finalize path is untouched.
    """
    rs = state.round_state
    counts: dict[str, int] = {}
    for target in rs.endorsements.values():
        counts[target] = counts.get(target, 0) + 1
    if not counts:
        return None, 0
    top = max(counts.values())
    tied = [k for k, c in counts.items() if c == top]
    winner = max(
        tied, key=lambda k: (rs.borda_scores.get(k, 0), -state.active_models.index(k))
    )
    return winner, top


def _positions_frozen(state: DebateState) -> bool:
    """True if this round's votes + endorsements exactly match the previous round's.

    The models are no longer moving, so more rounds won't help — time to stop.
    """
    if len(state.rounds) < 2:
        return False
    cur, prev = state.rounds[-1], state.rounds[-2]
    return cur.votes == prev.votes and cur.endorsements == prev.endorsements


def determine_consensus(state: DebateState) -> tuple[str, str]:
    counts = tally_votes(state)
    if not counts:
        return "wait", "No votes yet"

    rs = state.round_state
    total = len(state.active_models)
    majority = state.majority()
    winner, top = _endorsement_tally(state)

    # Consensus = a majority endorsing the *same* proposal (terminal). A bare
    # majority of FINALIZE votes is NOT enough on its own.
    if winner is not None and top >= majority:
        rs.consensus_winner = winner
        scope = "Unanimous" if top == total else f"Majority ({top}/{total})"
        return "finalize", f"{scope} endorsement of {provider_name(winner)}'s proposal."

    # There is no fixed round cap — the debate runs until the models converge or
    # stop moving. ``max_rounds`` is only a high safety fuse against runaway.
    if state.max_rounds and state.current_round >= state.max_rounds:
        if winner is not None:
            rs.consensus_winner = winner
        return "deadlocked", f"Safety fuse: round limit ({state.max_rounds}) reached."

    # The models have frozen into the same standoff as last round → stop.
    if _positions_frozen(state):
        if winner is not None:
            rs.consensus_winner = winner
        return "deadlocked", "Stable disagreement — positions unchanged across two rounds."

    # Otherwise let them keep deliberating: a revise majority states a focus;
    # anything else is an unsettled split that goes another round.
    if counts.get("revise", 0) >= majority:
        return "revise", f"Majority vote to revise ({counts['revise']}/{total})."

    return "split", f"No consensus. Votes: {counts}. Continuing."


def _extract_revise_focus(state: DebateState) -> str:
    rs = state.round_state
    for name, v in rs.votes.items():
        if v == Vote.REVISE:
            out = rs.model_outputs.get(name)
            if out and out.vote_reasoning and "REVISE:" in out.vote_reasoning.upper():
                m = re.search(r"REVISE:\s*(.+)", out.vote_reasoning, re.IGNORECASE)
                if m:
                    return m.group(1).strip()[:200]
    return "Refine proposals based on reviews"


def render_final_answer(state: DebateState) -> str:
    rs = state.round_state
    winner, top = _endorsement_tally(state)
    n_active = len(state.active_models)
    if state.status == "done":
        title = "# Final Consensus"
        status_line = (
            f"{top}/{n_active} endorsed {provider_name(winner)}'s proposal"
            if winner else "consensus reached"
        )
    elif state.status == "over_budget":
        title = "# Debate Result (stopped — budget)"
        status_line = rs.consensus_reason or "budget exceeded"
    else:
        title = "# Debate Result (no consensus)"
        status_line = f"deadlocked — final votes: {tally_votes(state)}"
    lines = [
        title,
        "",
        f"**Prompt:** {state.prompt}",
        f"**Rounds:** {state.current_round}",
        f"**Outcome:** {status_line}",
    ]
    if state.dropped_models:
        dropped = ", ".join(
            f"{provider_name(n)} ({why})" for n, why in state.dropped_models.items()
        )
        lines.append(f"**Dropped providers:** {dropped}")

    # Lead with the group-confirmed synthesis when one was accepted; the verbatim
    # proposals still follow below for full auditability.
    win_out = rs.model_outputs.get(rs.consensus_winner) if rs.consensus_winner else None
    if state.status == "done" and rs.synthesis_used and win_out and win_out.synthesis:
        tally = rs.confirm_tally or {}
        approve = tally.get("APPROVE", 0)
        lines += [
            "",
            "## Synthesis (group-confirmed)",
            "",
            f"*Merged from all proposals; confirmed by {approve}/{n_active}.*",
            "",
            win_out.synthesis,
            "",
        ]

    if win_out and win_out.proposal:
        wname = provider_name(rs.consensus_winner)
        endorsers = [
            provider_name(v) for v, t in rs.endorsements.items() if t == rs.consensus_winner
        ]
        head = (
            "## Consensus Answer" if state.status == "done"
            else "## Best-Effort Answer (no consensus)"
        )
        attribution = f"*{wname} — endorsed by {len(endorsers)}/{n_active}"
        attribution += (f": {', '.join(endorsers)}*" if endorsers else "*")
        lines += ["", head, "", attribution, "", win_out.proposal, ""]

    lines += ["", "## All Proposals", ""]
    for name in state.active_models:
        out = rs.model_outputs.get(name)
        if not out or not out.proposal:
            continue
        pname = provider_name(name)
        vote = rs.votes.get(name)
        vote_str = vote.value if isinstance(vote, Vote) else "—"
        endorsed = rs.endorsements.get(name)
        endorsed_str = f" → endorsed {provider_name(endorsed)}" if endorsed else ""
        lines.append(f"### {pname} (voted: {vote_str}{endorsed_str})")
        lines.append("")
        lines.append(out.proposal)
        lines.append("")

    if rs.borda_scores:
        ranked = sorted(rs.borda_scores.items(), key=lambda kv: kv[1], reverse=True)
        lines += ["## Ranking (Borda)", ""]
        for name, score in ranked:
            lines.append(f"- {provider_name(name)}: {score}")
        lines.append("")

    if state.usage:
        lines.append("## Cost")
        lines.append("")
        lines.append("| Model | Calls | Input | Output | Cached | Est. cost |")
        lines.append("|---|---|---|---|---|---|")
        for name in state.active_models:
            u = state.usage.get(name)
            if not u:
                continue
            c = f"${u.cost:.4f}" + ("" if u.cost_known else "+?")
            lines.append(
                f"| {provider_name(name)} | {u.calls} | {u.input_tokens} | "
                f"{u.output_tokens} | {u.cached_tokens} | {c} |"
            )
        total = f"${state.total_cost():.4f}" + ("" if state.cost_known() else " + unknown")
        lines.append("")
        lines.append(f"**Total estimated cost:** {total}")
        lines.append("")

    if state.sources:
        lines.append("## Sources")
        lines.append("")
        for i, s in enumerate(state.sources, 1):
            lines.append(f"{i}. [{s.title or s.url}]({s.url})")
        lines.append("")

    lines.append("---")
    lines.append("*Consensus via Ensemble — multi-model filesystem debate protocol.*")
    return "\n".join(lines)


def write_final_answer(state: DebateState, debate_dir: Path) -> None:
    text = render_final_answer(state)
    state.final_answer = text
    atomic_write_text(debate_dir / "final.md", text)


def _finalize(state: DebateState, debate_dir: Path, *, synthesis: bool) -> str:
    """Conclude a reached consensus, optionally using the confirmed synthesis.

    ``synthesis=False`` reproduces today's verbatim-winner output exactly; it is
    the fallback for any failure or stall in the post-consensus phases, where the
    decision is already made and must not be undone.
    """
    rs = state.round_state
    rs.synthesis_used = synthesis
    if synthesis:
        rs.synthesis_author = rs.consensus_winner
    state.status = "done"
    state.current_phase = Phase.DONE
    write_final_answer(state, debate_dir)
    save_state(state, debate_dir / "state.json")
    return "done"


# --- Coordinator loop --------------------------------------------------------


async def coordinator_loop(
    debate_dir: Path,
    poll_interval: float = 1.0,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
) -> str:
    state_path = debate_dir / "state.json"
    handled_phases: set[str] = set()
    last_progress = time.monotonic()

    while True:
        # Stall guard runs first, so a missing/unreadable state file can't make
        # the loop (and the agents waiting on it) spin forever.
        stalled = time.monotonic() - last_progress > stall_timeout

        if not state_path.exists():
            if stalled:
                log.error("No state file after %.0fs — giving up", stall_timeout)
                return "deadlocked"
            await asyncio.sleep(poll_interval)
            continue

        try:
            state = load_state(state_path)
        except Exception as e:
            if stalled:
                log.error("State unreadable for %.0fs (%s) — giving up", stall_timeout, e)
                return "deadlocked"
            log.debug("state read failed (%s), retrying", e)
            await asyncio.sleep(poll_interval)
            continue

        if state.is_finished:
            log.info("Debate finished. Status: %s", state.status)
            return state.status

        rs = state.round_state
        phase_key = f"r{state.current_round}-{rs.phase.value}"

        # A provider whose agent gave up leaves a `.failed` sentinel. Drop it so
        # the phase can complete on the survivors — unless that would leave too
        # few models, in which case end gracefully now instead of stalling.
        failed = failed_models(debate_dir, state)
        if failed and rs.phase in (Phase.SYNTHESIS, Phase.CONFIRM):
            # Consensus is already reached; a failure in the optional post-consensus
            # phases must ship the verbatim answer, never undo the decision.
            log.warning("Failure during %s — finalizing verbatim", rs.phase.value)
            salvage_phase_outputs(debate_dir, state)
            return _finalize(state, debate_dir, synthesis=False)
        if failed:
            for name in failed:
                state.dropped_models.setdefault(name, failure_reason(debate_dir, state, name))
            survivors = [m for m in state.active_models if m not in failed]
            if len(survivors) < MIN_MODELS:
                reason = (f"Only {len(survivors)} live provider(s); need {MIN_MODELS}. "
                          f"Dropped: {', '.join(failed)}.")
                log.error(reason)
                salvage_phase_outputs(debate_dir, state)
                _record_consensus(state, "deadlocked", reason)
                state.status = "deadlocked"
                state.current_phase = Phase.DEADLOCKED
                write_final_answer(state, debate_dir)
                save_state(state, state_path)
                return "deadlocked"
            log.warning("Dropping unresponsive provider(s) %s; continuing with %s",
                        ", ".join(failed), ", ".join(survivors))
            state.active_models = survivors
            last_progress = time.monotonic()  # topology changed — that's progress
            save_state(state, state_path)
            continue

        if phase_key in handled_phases or not all_phase_complete(debate_dir, state):
            if time.monotonic() - last_progress > stall_timeout:
                # A stall in the post-consensus phases finalizes verbatim — the
                # decision stands; only the optional synthesis is forgone.
                if rs.phase in (Phase.SYNTHESIS, Phase.CONFIRM):
                    log.warning("Stalled in %s — finalizing verbatim", rs.phase.value)
                    salvage_phase_outputs(debate_dir, state)
                    return _finalize(state, debate_dir, synthesis=False)
                log.error("Stalled %.0fs with no progress — declaring deadlock", stall_timeout)
                salvage_phase_outputs(debate_dir, state)
                _record_consensus(state, "deadlocked",
                                   f"Stalled in {rs.phase.value} (round {state.current_round}).")
                state.status = "deadlocked"
                state.current_phase = Phase.DEADLOCKED
                write_final_answer(state, debate_dir)
                save_state(state, state_path)
                return "deadlocked"
            await asyncio.sleep(poll_interval)
            continue

        # A phase just completed: fold each model's contribution into the
        # accumulated per-round outputs (merge, never replace).
        for name, output in read_phase_outputs(debate_dir, state).items():
            rs.model_outputs.setdefault(name, ModelOutput()).merge_from(output)

        accumulate_usage(debate_dir, state)
        handled_phases.add(phase_key)
        last_progress = time.monotonic()

        # Stop before spending more if the budget is blown.
        if state.budget is not None and state.total_cost() >= state.budget:
            _record_consensus(
                state, "over_budget",
                f"Budget ${state.budget:.4f} reached (spent ${state.total_cost():.4f}).",
            )
            state.status = "over_budget"
            log.error("Budget exceeded: spent $%.4f of $%.4f", state.total_cost(), state.budget)
            write_final_answer(state, debate_dir)
            save_state(state, state_path)
            return "over_budget"

        if rs.phase == Phase.PROPOSING:
            log.info("Round %d: proposals received → REVIEWING", state.current_round)
            _set_phase(state, Phase.REVIEWING)
            save_state(state, state_path)

        elif rs.phase == Phase.REVIEWING:
            log.info("Round %d: reviews received → REBUTTAL", state.current_round)
            _set_phase(state, Phase.REBUTTAL)
            save_state(state, state_path)

        elif rs.phase == Phase.REBUTTAL:
            log.info("Round %d: rebuttals received → VOTING", state.current_round)
            _set_phase(state, Phase.VOTING)
            save_state(state, state_path)

        elif rs.phase == Phase.VOTING:
            for name, out in rs.model_outputs.items():
                if out.vote is not None:
                    rs.votes[name] = out.vote
                if out.ranking:
                    rs.rankings[name] = out.ranking
            _record_endorsements(state)
            rs.borda_scores = compute_borda(state)

            action, reason = determine_consensus(state)
            rs.consensus_action = action
            rs.consensus_reason = reason
            log.info("Round %d: votes %s → %s (%s)",
                     state.current_round, dict(rs.votes), action.upper(), reason)

            if action == "finalize":
                # Consensus reached: begin synthesis-as-candidate instead of
                # finishing immediately. consensus_winner was set in
                # determine_consensus; the loop now drives SYNTHESIS → CONFIRM.
                _set_phase(state, Phase.SYNTHESIS)
                save_state(state, state_path)
                await asyncio.sleep(poll_interval)
                continue

            if action == "deadlocked":
                state.status = "deadlocked"
                state.current_phase = Phase.DEADLOCKED
                write_final_answer(state, debate_dir)
                save_state(state, state_path)
                return "deadlocked"

            # revise / split → next round
            next_round = state.current_round + 1
            focus = _extract_revise_focus(state) if action == "revise" else ""
            state.rounds.append(
                RoundState(round_num=next_round, phase=Phase.PROPOSING, focus=focus)
            )
            state.current_round = next_round
            state.current_phase = Phase.PROPOSING
            log.info("Round %d starting (focus: %s)", next_round, focus or "—")
            save_state(state, state_path)

        elif rs.phase == Phase.SYNTHESIS:
            log.info("Round %d: synthesis written → CONFIRM", state.current_round)
            _set_phase(state, Phase.CONFIRM)
            save_state(state, state_path)

        elif rs.phase == Phase.CONFIRM:
            approve = sum(1 for o in rs.model_outputs.values() if o.confirm == "APPROVE")
            reject = sum(1 for o in rs.model_outputs.values() if o.confirm == "REJECT")
            rs.confirm_tally = {"APPROVE": approve, "REJECT": reject}
            used = approve >= state.majority()
            log.info("Round %d: confirm %s → synthesis %s",
                     state.current_round, rs.confirm_tally,
                     "ACCEPTED" if used else "rejected (verbatim)")
            return _finalize(state, debate_dir, synthesis=used)

        await asyncio.sleep(poll_interval)


def _set_phase(state: DebateState, phase: Phase) -> None:
    state.round_state.phase = phase
    state.current_phase = phase


def _record_endorsements(state: DebateState) -> None:
    """Resolve each FINALIZE voter's endorsed proposal into ``rs.endorsements``."""
    rs = state.round_state
    for name, out in rs.model_outputs.items():
        if out.vote == Vote.FINALIZE:
            rs.endorsements[name] = resolve_endorsement(out.vote_reasoning, state, name)


def _record_consensus(state: DebateState, action: str, reason: str) -> None:
    rs = state.round_state
    rs.consensus_action = action
    rs.consensus_reason = reason
    for name, out in rs.model_outputs.items():
        if out.vote is not None:
            rs.votes[name] = out.vote
    _record_endorsements(state)
