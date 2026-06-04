"""Shared debate orchestration used by the CLI, the chat REPL, and the MCP server.

Keeping build + run in one place means every entry point creates and drives a
debate identically.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .agent import agent_loop
from .coordinator import (
    DEFAULT_STALL_TIMEOUT,
    coordinator_loop,
    create_debate,
    setup_debate_dir,
)
from .models import Usage, default_model_id, get_api_key, usage_cost
from .search import web_search
from .state import MIN_MODELS, DebateState, assign_aliases, save_state

log = logging.getLogger("ensemble.orchestrator")

# Canonical provider order for display and selection.
DEFAULT_ORDER: list[str] = ["gpt4o", "claude", "deepseek"]
# High safety fuse on rounds — the debate normally ends by consensus or stable
# disagreement, not by hitting this.
DEFAULT_ROUNDS = 50

# Named stances for the optional anti-groupthink roles.
STANCE_LIBRARY: dict[str, str] = {
    "skeptic": (
        "Take a skeptical, critical stance. Stress-test every claim, surface "
        "risks, edge cases, and failure modes; accept nothing without justification."
    ),
    "advocate": (
        "Take an optimistic, constructive stance. Build the strongest possible "
        "case for the most promising approach."
    ),
    "pragmatist": (
        "Take a pragmatic stance. Favor what is simplest, most reliable, and "
        "shippable; weigh cost, complexity, and maintenance."
    ),
    "neutral": "Remain neutral and balanced; weigh all sides strictly on the merits.",
}

ROLE_PRESETS = ("none", "diverse", "redteam")


def assign_roles(
    active_models: list[str], preset: str = "none", overrides: dict[str, str] | None = None
) -> dict[str, str]:
    """Resolve a role preset + per-model overrides into stance instructions.

    Overrides may name a stance from STANCE_LIBRARY or provide custom text.
    """
    roles: dict[str, str] = {}
    if preset == "diverse":
        cycle = ["skeptic", "advocate", "pragmatist"]
        for i, m in enumerate(active_models):
            roles[m] = STANCE_LIBRARY[cycle[i % len(cycle)]]
    elif preset == "redteam":
        for i, m in enumerate(active_models):
            roles[m] = STANCE_LIBRARY["advocate" if i == 0 else "skeptic"]
    for m, stance in (overrides or {}).items():
        roles[m] = STANCE_LIBRARY.get(stance, stance)
        if stance not in STANCE_LIBRARY and len(stance) > 500:
            raise ValueError(
                f"Stance text for {m!r} exceeds 500 chars ({len(stance)}). "
                "Use a named stance or a shorter custom text."
            )
    return roles


def available_models(requested: list[str] | None = None) -> list[str]:
    """Of the requested providers (or all), those that have an API key set."""
    names = requested or DEFAULT_ORDER
    return [n for n in names if get_api_key(n)]


def build_debate(
    prompt: str,
    *,
    models: list[str] | None = None,
    rounds: int = DEFAULT_ROUNDS,
    quick: bool = False,
    overrides: dict[str, str] | None = None,
    budget: float | None = None,
    roles: dict[str, str] | None = None,
    grounding: bool = False,
) -> tuple[DebateState, Path]:
    """Create a debate's state + on-disk folder. Raises ValueError if too few keys.

    ``quick`` runs a single round (propose → review → vote, then finalize or
    deadlock) for low latency. ``deep`` (quick=False) allows up to ``rounds``.
    ``budget`` caps estimated USD spend; ``roles`` assigns per-model stances;
    ``grounding`` enables a web-search step before the debate.
    """
    if len(prompt) > 16000:
        raise ValueError(f"Prompt too long ({len(prompt)} chars; max 16000).")
    active = available_models(models)
    if len(active) < MIN_MODELS:
        raise ValueError(
            f"Need at least {MIN_MODELS} providers with API keys "
            "(OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY). "
            f"Available: {active or 'none'}."
        )
    overrides = overrides or {}
    state = create_debate(prompt)
    state.max_rounds = 1 if quick else rounds
    state.active_models = active
    state.participant_aliases = assign_aliases(active)
    state.model_ids = {n: overrides.get(n, default_model_id(n)) for n in active}
    state.budget = budget
    state.roles = {m: r for m, r in (roles or {}).items() if m in active}
    state.grounding = grounding

    if budget is not None:
        # The budget is enforced on *known* pricing; warn if a model's cost
        # can't be estimated (e.g. a custom model id) so the cap isn't silently
        # ineffective for it.
        probe = Usage(1, 0, 0)
        unpriced = [n for n in active if usage_cost(n, state.model_ids[n], probe) is None]
        if unpriced:
            log.warning(
                "budget set but pricing is unknown for %s — the cap cannot be "
                "enforced for those models", ", ".join(unpriced),
            )

    debate_dir = setup_debate_dir(state)
    return state, debate_dir


async def ground_debate(state: DebateState, debate_dir: Path) -> None:
    """If grounding is on and not already done, fetch sources and persist them."""
    if not state.grounding or state.sources:
        return
    state.sources = await web_search(state.prompt)
    save_state(state, debate_dir / "state.json")


async def run_debate(
    state: DebateState,
    debate_dir: Path,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    poll_interval: float | None = None,
) -> str:
    """Run agents + coordinator to completion. Returns final status string.

    ``poll_interval`` overrides the loops' polling cadence (mainly for tests);
    None keeps each loop's production default.
    """
    await ground_debate(state, debate_dir)
    a_kwargs = {} if poll_interval is None else {"poll_interval": poll_interval}
    c_kwargs: dict = {"stall_timeout": stall_timeout}
    if poll_interval is not None:
        c_kwargs["poll_interval"] = poll_interval
    agents = [
        asyncio.create_task(agent_loop(n, debate_dir, **a_kwargs)) for n in state.active_models
    ]
    coord = asyncio.create_task(coordinator_loop(debate_dir, **c_kwargs))
    try:
        status = await coord
    finally:
        for a in agents:
            a.cancel()
        await asyncio.gather(*agents, return_exceptions=True)
    return status
