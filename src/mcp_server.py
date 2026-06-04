"""MCP server exposing Ensemble as a callable tool.

Lets any MCP client (Claude Code, Cursor, Cline, Continue, Kilo Code, …) summon
a multi-model council from inside its own workflow. Bring your own API keys via
environment variables; at least two of OPENAI_API_KEY / ANTHROPIC_API_KEY /
DEEPSEEK_API_KEY must be set.

Run:  ensemble-mcp        (after `pip install "ensemble[mcp]"`)
"""
from __future__ import annotations

from .coordinator import DEFAULT_STALL_TIMEOUT
from .orchestrator import available_models, build_debate, run_debate
from .state import load_state

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover - import-time guard
    raise SystemExit(
        "The MCP SDK is not installed. Install it with:  pip install 'ensemble[mcp]'"
    ) from e

mcp = FastMCP("ensemble")


async def run_ensemble_debate(
    prompt: str,
    quick: bool = True,
    rounds: int = 50,
    models: list[str] | None = None,
    ground: bool = False,
    budget: float | None = None,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    poll_interval: float | None = None,
) -> str:
    """Core implementation (kept separate from the tool wrapper for testing)."""
    active = available_models(models)
    if len(active) < 2:
        return (
            "ERROR: Ensemble needs at least 2 providers with API keys set "
            "(OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY). "
            f"Currently available: {active or 'none'}."
        )

    state, debate_dir = build_debate(
        prompt, models=models, rounds=rounds, quick=quick, grounding=ground, budget=budget
    )
    status = await run_debate(
        state, debate_dir, stall_timeout=stall_timeout, poll_interval=poll_interval
    )
    final = load_state(debate_dir / "state.json")

    cost = f" · ~${final.total_cost():.4f}" if final.usage else ""
    synth = ""
    if status == "done" and final.rounds:
        synth = (" · synthesis=adopted" if final.round_state.synthesis_used
                 else " · synthesis=verbatim")
    header = (
        f"[ensemble] outcome={status} · models={', '.join(final.active_models)} · "
        f"rounds={final.current_round}{cost}{synth} · id={final.debate_id}\n\n"
    )
    return header + (final.final_answer or "(no final answer was produced)")


@mcp.tool()
async def ensemble_debate(
    prompt: str,
    quick: bool = True,
    rounds: int = 50,
    models: list[str] | None = None,
    ground: bool = False,
    budget: float | None = None,
) -> str:
    """Convene a council of multiple LLMs that independently propose an answer,
    peer-review each other (anonymously), rebut, and vote — iterating until they
    converge (or provably can't). On consensus the endorsed author drafts a
    merged answer the group confirms by vote. Returns the consensus document
    with each model's contribution, the final vote, a Borda ranking, an estimated
    cost breakdown, and any sources.

    Use this for high-stakes or contested questions where one model's answer
    isn't trustworthy enough on its own: architecture decisions, tradeoff
    analysis, code review, or research/fact verification. It is slower than a
    single model, so reserve it for the calls that warrant a second (and third)
    opinion rather than routine turns.

    Args:
        prompt: The question or proposal for the council to debate.
        quick: If true (default), run a single round for lower latency. Set
            false for a full multi-round debate (more thorough, slower).
        rounds: Safety fuse on rounds when quick is false; the debate normally
            ends by consensus or stable disagreement well before this.
        models: Subset of providers to use (any of "gpt4o", "claude",
            "deepseek"). Defaults to every provider that has an API key set.
        ground: If true, run a web search first and have models cite sources
            (requires TAVILY_API_KEY; ignored if unset).
        budget: Optional cap on estimated USD spend; the debate stops if hit.
    """
    return await run_ensemble_debate(
        prompt, quick=quick, rounds=rounds, models=models, ground=ground, budget=budget
    )


def main() -> None:
    """Entry point for the `ensemble-mcp` script (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
