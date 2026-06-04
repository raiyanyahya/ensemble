from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Phase(StrEnum):
    PROPOSING = "proposing"
    REVIEWING = "reviewing"
    REBUTTAL = "rebuttal"
    VOTING = "voting"
    SYNTHESIS = "synthesis"
    CONFIRM = "confirm"
    DONE = "done"
    DEADLOCKED = "deadlocked"


class Vote(StrEnum):
    FINALIZE = "finalize"
    REVISE = "revise"
    SPLIT = "split"


# The file suffix each phase's contribution is written under, and the
# ModelOutput field that phase populates. Keeping these in one place means the
# agent (writer) and coordinator (reader) can never disagree about layout.
PHASE_FILE_SUFFIX: dict[Phase, str] = {
    Phase.PROPOSING: "proposal",
    Phase.REVIEWING: "review",
    Phase.REBUTTAL: "rebuttal",
    Phase.VOTING: "vote",
    Phase.SYNTHESIS: "synthesis",
    Phase.CONFIRM: "confirm",
}

# A debate needs at least this many live models to be meaningful. Enforced both
# up front (build_debate) and mid-debate when dropping unresponsive providers.
MIN_MODELS = 2


def assign_aliases(active_models: list[str]) -> dict[str, str]:
    """Map each model to a neutral, identity-free label (Participant A, B, …).

    Used only when building prompts *for* the models, so they judge each other's
    arguments on merit rather than on brand. User-facing output keeps real names.
    """
    return {m: f"Participant {chr(ord('A') + i)}" for i, m in enumerate(active_models)}
class ModelUsage(BaseModel):
    """Accumulated token usage and estimated cost for one model in a debate."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    cost: float = 0.0          # estimated USD; only counts calls with known pricing
    cost_known: bool = True    # False once a call had unknown pricing (custom model)

    def add(self, input_t: int, output_t: int, cached_t: int, cost: float | None) -> None:
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.cached_tokens += cached_t
        self.calls += 1
        if cost is None:
            self.cost_known = False
        else:
            self.cost += cost


class Source(BaseModel):
    """A web result injected into the debate for grounding."""
    title: str = ""
    url: str = ""
    snippet: str = ""


class ModelOutput(BaseModel):
    proposal: str = ""
    reviews: str = ""
    rebuttal: str = ""
    vote: Vote | None = None
    vote_reasoning: str = ""
    # Optional Borda ballot (participant labels, best→worst) emitted alongside a vote.
    ranking: list[str] = Field(default_factory=list)
    # Post-consensus synthesis-as-candidate phases.
    synthesis: str = ""
    confirm: str = ""  # "APPROVE" / "REJECT" / ""

    def has_phase(self, phase: Phase) -> bool:
        """Whether this output already carries a contribution for ``phase``."""
        if phase == Phase.PROPOSING:
            return bool(self.proposal)
        if phase == Phase.REVIEWING:
            return bool(self.reviews)
        if phase == Phase.REBUTTAL:
            return bool(self.rebuttal)
        if phase == Phase.VOTING:
            return self.vote is not None
        if phase == Phase.SYNTHESIS:
            return bool(self.synthesis)
        if phase == Phase.CONFIRM:
            return bool(self.confirm)
        return True

    def for_phase(self, phase: Phase) -> ModelOutput:
        """A copy carrying only the field(s) that ``phase`` is responsible for.

        Models sometimes emit a section out of turn — e.g. a ``## Vote`` inside a
        review. Harvesting only the current phase's field stops that leaked vote
        from being recorded early (which would make the model skip its real vote
        and stall the debate).
        """
        if phase == Phase.PROPOSING:
            return ModelOutput(proposal=self.proposal)
        if phase == Phase.REVIEWING:
            return ModelOutput(reviews=self.reviews)
        if phase == Phase.REBUTTAL:
            return ModelOutput(rebuttal=self.rebuttal)
        if phase == Phase.VOTING:
            return ModelOutput(
                vote=self.vote, vote_reasoning=self.vote_reasoning, ranking=self.ranking
            )
        if phase == Phase.SYNTHESIS:
            return ModelOutput(synthesis=self.synthesis)
        if phase == Phase.CONFIRM:
            return ModelOutput(confirm=self.confirm)
        return ModelOutput()

    def merge_from(self, other: ModelOutput) -> None:
        """Fold another output's non-empty fields into this one in place.

        This is the fix for cross-phase data loss: a model contributes a
        different field each phase, and we accumulate rather than replace.
        """
        if other.proposal:
            self.proposal = other.proposal
        if other.reviews:
            self.reviews = other.reviews
        if other.rebuttal:
            self.rebuttal = other.rebuttal
        if other.vote is not None:
            self.vote = other.vote
        if other.vote_reasoning:
            self.vote_reasoning = other.vote_reasoning
        if other.ranking:
            self.ranking = other.ranking
        if other.synthesis:
            self.synthesis = other.synthesis
        if other.confirm:
            self.confirm = other.confirm


class RoundState(BaseModel):
    round_num: int
    phase: Phase
    focus: str = ""
    model_outputs: dict[str, ModelOutput] = Field(default_factory=dict)
    votes: dict[str, Vote] = Field(default_factory=dict)
    # For FINALIZE voters: which proposal they endorse (voter key -> endorsed key).
    endorsements: dict[str, str] = Field(default_factory=dict)
    consensus_action: str | None = None
    consensus_reason: str = ""
    # The proposal that won (majority endorsement), or plurality on a deadlock.
    consensus_winner: str = ""
    # Borda ranking signal (additive): raw ballots and aggregated scores.
    rankings: dict[str, list[str]] = Field(default_factory=dict)
    borda_scores: dict[str, int] = Field(default_factory=dict)
    # Synthesis-as-candidate outcome (only set on the finalize path).
    synthesis_used: bool = False
    synthesis_author: str = ""
    confirm_tally: dict[str, int] = Field(default_factory=dict)


class DebateState(BaseModel):
    debate_id: str
    prompt: str
    created_at: str = ""
    updated_at: str = ""
    rounds: list[RoundState] = Field(default_factory=list)
    current_round: int = 0
    current_phase: Phase = Phase.PROPOSING
    status: str = "active"  # active | done | deadlocked
    final_answer: str = ""
    active_models: list[str] = Field(default_factory=lambda: ["gpt4o", "claude", "deepseek"])
    # Per-debate model id overrides, so a debate is reproducible on resume.
    model_ids: dict[str, str] = Field(default_factory=dict)
    max_rounds: int = 50  # high safety fuse, not a normal terminator (0 = unlimited)
    # Optional per-model stance/role instructions to fight groupthink.
    roles: dict[str, str] = Field(default_factory=dict)
    # Web-search grounding.
    grounding: bool = False
    sources: list[Source] = Field(default_factory=list)
    # Cost controls and accounting.
    budget: float | None = None  # USD; debate stops if exceeded
    usage: dict[str, ModelUsage] = Field(default_factory=dict)
    # Providers dropped mid-debate after their agent gave up, name -> reason.
    dropped_models: dict[str, str] = Field(default_factory=dict)
    # Identity-free labels shown to the models (model key -> "Participant X").
    participant_aliases: dict[str, str] = Field(default_factory=dict)

    @property
    def round_state(self) -> RoundState:
        return self.rounds[self.current_round - 1]

    @property
    def is_finished(self) -> bool:
        return self.status in ("done", "deadlocked", "over_budget")

    def majority(self) -> int:
        return len(self.active_models) // 2 + 1

    def total_cost(self) -> float:
        return sum(u.cost for u in self.usage.values())

    def cost_known(self) -> bool:
        return all(u.cost_known for u in self.usage.values())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state(state_path: Path) -> DebateState:
    return DebateState.model_validate_json(state_path.read_text())


def save_state(state: DebateState, state_path: Path) -> None:
    """Atomically persist state.

    Write to a temp file in the same directory, then ``os.replace`` it over the
    target. ``os.replace`` is atomic on POSIX and Windows, so a concurrent
    reader (the agents, polling) never observes a half-written file.
    """
    state.updated_at = now_iso()
    payload = state.model_dump_json(indent=2)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=state_path.parent, prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, state_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write a text file (used for model contribution files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
