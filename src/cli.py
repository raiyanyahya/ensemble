from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import typer
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .coordinator import DEBATES_DIR, DEFAULT_STALL_TIMEOUT
from .models import PROVIDERS, get_api_key
from .orchestrator import (
    DEFAULT_ORDER,
    DEFAULT_ROUNDS,
    ROLE_PRESETS,
    assign_roles,
    available_models,
    build_debate,
    run_debate,
)
from .search import search_enabled
from .state import DebateState, Phase, load_state
from .ui import console, model_style, setup_logging

app = typer.Typer(
    help="Multi-model consensus debate — top LLMs debate, review, and vote via the filesystem.",
    add_completion=False,
    no_args_is_help=True,
)

NO_KEYS_MSG = (
    "[bold red]Need at least 2 API keys.[/bold red] Set any two of:\n"
    "  OPENAI_API_KEY · ANTHROPIC_API_KEY · DEEPSEEK_API_KEY"
)


_DEBATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _resolve_debate_dir(debate_id: str) -> Path:
    """Map a user-supplied debate id to its folder, rejecting path traversal.

    Generated ids look like ``20260525-181144-ecc843``; anything with a slash,
    ``..``, or other unexpected characters is refused so the id can't escape
    DEBATES_DIR (e.g. ``show /etc`` or ``resume ../../x``).
    """
    if not _DEBATE_ID_RE.fullmatch(debate_id):
        console.print(f"[red]Invalid debate id: {debate_id!r}[/red]")
        raise typer.Exit(2)
    return DEBATES_DIR / debate_id


def _load_state_or_exit(state_path: Path, debate_id: str):
    try:
        return load_state(state_path)
    except Exception as e:
        console.print(f"[red]Could not read state for '{debate_id}': {e}[/red]")
        raise typer.Exit(1) from None


def _parse_model_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise typer.BadParameter(f"--model expects NAME=MODEL_ID, got {item!r}")
        name, model_id = item.split("=", 1)
        name = name.strip()
        if name not in PROVIDERS:
            raise typer.BadParameter(f"unknown provider {name!r}; choose from {list(PROVIDERS)}")
        overrides[name] = model_id.strip()
    return overrides


def _parse_role_overrides(values: list[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise typer.BadParameter(f"--role expects NAME=STANCE, got {item!r}")
        name, stance = item.split("=", 1)
        name = name.strip()
        if name not in PROVIDERS:
            raise typer.BadParameter(f"unknown provider {name!r}; choose from {list(PROVIDERS)}")
        roles[name] = stance.strip()
    return roles


def _resolve_setup(roles_preset: str, role: list[str], ground: bool):
    """Validate role flags + grounding, returning resolved roles dict."""
    if roles_preset not in ROLE_PRESETS:
        raise typer.BadParameter(f"--roles must be one of {list(ROLE_PRESETS)}")
    assigned = assign_roles(available_models(), roles_preset, _parse_role_overrides(role))
    if ground and not search_enabled():
        console.print("[yellow]--ground set but TAVILY_API_KEY is missing — "
                      "proceeding ungrounded.[/yellow]")
    return assigned


def _provider_table(model_ids: dict[str, str]) -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Model ID")
    table.add_column("Status")
    for name in DEFAULT_ORDER:
        p = PROVIDERS[name]
        active = get_api_key(name) is not None
        status = "[green]active[/green]" if active else "[dim]no API key[/dim]"
        table.add_row(
            f"[{model_style(name)}]{name}[/{model_style(name)}]",
            p.display,
            model_ids.get(name, p.default_model),
            status,
        )
    return table


def _progress_table(state: DebateState) -> Table:
    rs = state.round_state
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Proposal")
    table.add_column("Reviews")
    table.add_column("Vote")
    mark = {True: "[green]✓[/green]", False: "[dim]·[/dim]"}
    for name in state.active_models:
        out = rs.model_outputs.get(name)
        table.add_row(
            f"[{model_style(name)}]{name}[/{model_style(name)}]",
            mark[bool(out and out.proposal)],
            mark[bool(out and out.reviews)],
            out.vote.value if out and out.vote else "[dim]·[/dim]",
        )
    return table


def _phase_footer(state: DebateState) -> str | None:
    """A compact synthesis / confirm / Borda summary for the live panel and `status`.

    The per-model table covers the recurring propose→review→vote work; this footer
    surfaces the post-vote stages the table's columns can't — the winner-only
    synthesis, the confirm tally, and the Borda ranking — and only appears once a
    debate has actually produced them. Returns ``None`` when there's nothing yet.
    """
    rs = state.round_state
    lines: list[str] = []

    if rs.borda_scores and any(rs.borda_scores.values()):
        order = sorted(rs.borda_scores, key=lambda k: (-rs.borda_scores[k], k))
        ranked = " ▸ ".join(f"[{model_style(n)}]{n}[/{model_style(n)}]" for n in order)
        lines.append(f"[dim]ranking[/dim]   {ranked}")

    winner = rs.consensus_winner
    win_out = rs.model_outputs.get(winner) if winner else None
    in_synth = state.current_phase in (Phase.SYNTHESIS, Phase.CONFIRM)
    drafted = bool(win_out and win_out.synthesis)
    # Synthesis only runs on the finalize path — show it while it's happening, or
    # once it has (a deadlock sets a plurality winner but never synthesizes).
    if winner and (in_synth or drafted or rs.synthesis_used):
        wlabel = f"[{model_style(winner)}]{winner}[/{model_style(winner)}]"
        state_str = f"✓ by {wlabel}" if drafted else f"[dim]…[/dim] {wlabel} drafting"
        lines.append(f"[dim]synthesis[/dim] {state_str}")
        if rs.confirm_tally or in_synth:
            approve = (
                rs.confirm_tally.get("APPROVE", 0) if rs.confirm_tally
                else sum(1 for o in rs.model_outputs.values() if o.confirm == "APPROVE")
            )
            tail = ""
            if state.status == "done":
                tail = (" → [green]adopted[/green]" if rs.synthesis_used
                        else " → [yellow]verbatim winner[/yellow]")
            lines.append(f"[dim]confirm[/dim]   {approve}/{len(state.active_models)} approve{tail}")

    return "\n".join(lines) if lines else None


def _live_panel(debate_dir: Path):
    try:
        state = load_state(debate_dir / "state.json")
    except Exception:
        return Panel("[dim]starting…[/dim]", border_style="cyan")
    cost = state.total_cost()
    cost_str = f" · ~${cost:.4f}" if state.usage else ""
    budget_str = f"/${state.budget:.2f}" if state.budget else ""
    header = (
        f"[bold]⚖  council debating[/bold]  "
        f"[dim]round {state.current_round}/{state.max_rounds} · "
        f"phase {state.current_phase.value}{cost_str}{budget_str}[/dim]"
    )
    body = [header, _progress_table(state)]
    footer = _phase_footer(state)
    if footer:
        body.append(footer)
    return Panel(Group(*body), border_style="cyan")


async def _run_with_live(state: DebateState, debate_dir: Path, stall_timeout: float) -> str:
    task = asyncio.create_task(run_debate(state, debate_dir, stall_timeout))
    with Live(
        _live_panel(debate_dir), console=console, refresh_per_second=8, transient=True
    ) as live:
        while not task.done():
            live.update(_live_panel(debate_dir))
            await asyncio.sleep(0.15)
        live.update(_live_panel(debate_dir))
    return await task


def _finish_banner(debate_dir: Path, status: str) -> None:
    final_file = debate_dir / "final.md"
    if status == "done":
        console.print(Panel("[bold green]Consensus reached[/bold green]", expand=False))
    else:
        console.print(Panel(f"[bold yellow]No consensus — {status}[/bold yellow]", expand=False))
    if final_file.exists():
        console.print(Markdown(final_file.read_text()))
    console.print(f"\n[dim]Artifacts:[/dim] {debate_dir}")


@app.command()
def debate(
    prompt: str = typer.Argument(..., help="The question or topic for the models to debate"),
    rounds: int = typer.Option(
        DEFAULT_ROUNDS, "--rounds", "-r", min=1, max=100,
        help="Safety fuse on rounds; the debate ends by consensus or stable disagreement first",
    ),
    quick: bool = typer.Option(False, "--quick", "-q", help="Single round for speed"),
    budget: float | None = typer.Option(
        None, "--budget", "-b", help="Stop once estimated USD spend reaches this"
    ),
    ground: bool = typer.Option(
        False, "--ground", help="Web-search grounding (needs TAVILY_API_KEY)"
    ),
    roles: str = typer.Option("none", "--roles", help="Stance preset: none | diverse | redteam"),
    role: list[str] = typer.Option(
        [], "--role", help="Per-model stance, e.g. --role gpt4o=skeptic"
    ),
    model: list[str] = typer.Option(
        [], "--model", "-m", help="Override a model id, e.g. -m claude=claude-sonnet-4-6"
    ),
    stall_timeout: float = typer.Option(
        DEFAULT_STALL_TIMEOUT, "--stall-timeout", help="Seconds of no progress before deadlock"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Start a new debate and run it to consensus (or deadlock)."""
    setup_logging(verbose)
    overrides = _parse_model_overrides(model)
    assigned_roles = _resolve_setup(roles, role, ground)

    if len(available_models()) < 2:
        console.print(NO_KEYS_MSG)
        raise typer.Exit(1)

    state, debate_dir = build_debate(
        prompt, rounds=rounds, quick=quick, overrides=overrides,
        budget=budget, roles=assigned_roles, grounding=ground,
    )

    console.print(Panel.fit(
        f"[bold]Debate[/bold] {state.debate_id}\n[dim]{prompt}[/dim]",
        border_style="blue",
    ))
    console.print(_provider_table(state.model_ids))
    mode = "quick (1 round)" if quick else f"up to {rounds} rounds"
    extras = []
    if budget:
        extras.append(f"budget ${budget:.2f}")
    if state.roles:
        extras.append(f"roles: {roles}")
    if state.grounding:
        extras.append("grounded")
    extra_str = ("   [dim]" + " · ".join(extras) + "[/dim]") if extras else ""
    console.print(f"[dim]Mode:[/dim] {mode}{extra_str}   [dim]Folder:[/dim] {debate_dir}\n")

    try:
        status = asyncio.run(run_debate(state, debate_dir, stall_timeout))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Resume with:[/yellow] "
                      f"ensemble resume {state.debate_id}")
        raise typer.Exit(130) from None

    _finish_banner(debate_dir, status)


@app.command()
def chat(
    quick: bool = typer.Option(True, "--quick/--deep", help="Default per-question depth"),
    rounds: int = typer.Option(
        DEFAULT_ROUNDS, "--rounds", "-r", min=1, max=100,
        help="Safety fuse on rounds in deep mode (ends by consensus/stable disagreement first)",
    ),
    budget: float | None = typer.Option(None, "--budget", "-b", help="USD cap per question"),
    ground: bool = typer.Option(
        False, "--ground", help="Web-search grounding (needs TAVILY_API_KEY)"
    ),
    roles: str = typer.Option("none", "--roles", help="Stance preset: none | diverse | redteam"),
    role: list[str] = typer.Option(
        [], "--role", help="Per-model stance, e.g. --role gpt4o=skeptic"
    ),
    model: list[str] = typer.Option([], "--model", "-m", help="Override a model id"),
    stall_timeout: float = typer.Option(DEFAULT_STALL_TIMEOUT, "--stall-timeout"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive session — ask questions and the council debates each one.

    Commands: /quick /deep /rounds N /list /help /exit
    """
    setup_logging(verbose)
    if not verbose:
        logging.getLogger("ensemble").setLevel(logging.ERROR)  # keep the live panel clean
    overrides = _parse_model_overrides(model)
    assigned_roles = _resolve_setup(roles, role, ground)

    if len(available_models()) < 2:
        console.print(NO_KEYS_MSG)
        raise typer.Exit(1)

    deep = not quick
    cur_rounds = rounds
    flags = []
    if budget:
        flags.append(f"budget ${budget:.2f}/q")
    if roles != "none" or role:
        flags.append(f"roles: {roles}")
    if ground:
        flags.append("grounded")
    flag_str = f"\n[dim]{' · '.join(flags)}[/dim]" if flags else ""
    console.print(Panel.fit(
        "[bold]Ensemble chat[/bold] — a council of LLMs debates each question.\n"
        f"[dim]models: {', '.join(available_models())}[/dim]" + flag_str + "\n"
        "[dim]/quick · /deep · /rounds N · /list · /help · /exit[/dim]",
        border_style="blue",
    ))

    while True:
        try:
            q = Prompt.ask("\n[bold cyan]you[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        low = q.lower()
        if low in ("/exit", "/quit", "exit", "quit", ":q"):
            break
        if low in ("/help", "help", "?"):
            console.print(
                "[dim]/quick[/dim] one round · [dim]/deep[/dim] multi-round · "
                "[dim]/rounds N[/dim] set cap · [dim]/list[/dim] past debates · [dim]/exit[/dim]"
            )
            continue
        if low == "/deep":
            deep = True
            console.print("[dim]mode → deep[/dim]")
            continue
        if low == "/quick":
            deep = False
            console.print("[dim]mode → quick[/dim]")
            continue
        if low.startswith("/rounds"):
            parts = q.split()
            if len(parts) == 2 and parts[1].isdigit():
                cur_rounds = max(1, min(100, int(parts[1])))
                console.print(f"[dim]round fuse → {cur_rounds}[/dim]")
            else:
                console.print("[dim]usage: /rounds N[/dim]")
            continue
        if low == "/list":
            list_debates()
            continue

        try:
            state, debate_dir = build_debate(
                q, rounds=cur_rounds, quick=not deep, overrides=overrides,
                budget=budget, roles=assigned_roles, grounding=ground,
            )
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            continue

        mode = "deep" if deep else "quick"
        console.print(f"[dim]{mode} · {', '.join(state.active_models)} · {state.debate_id}[/dim]")
        try:
            status = asyncio.run(_run_with_live(state, debate_dir, stall_timeout))
        except KeyboardInterrupt:
            console.print("[yellow]debate interrupted[/yellow]")
            continue

        final_file = debate_dir / "final.md"
        if final_file.exists():
            console.print(Markdown(final_file.read_text()))
        tag = "[green]consensus[/green]" if status == "done" else f"[yellow]{status}[/yellow]"
        console.print(f"{tag}  [dim]{debate_dir}[/dim]")

    console.print("\n[dim]bye.[/dim]")


@app.command()
def resume(
    debate_id: str = typer.Argument(..., help="Debate ID to resume"),
    stall_timeout: float = typer.Option(DEFAULT_STALL_TIMEOUT, "--stall-timeout"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Resume an unfinished debate from its saved state."""
    setup_logging(verbose)
    debate_dir = _resolve_debate_dir(debate_id)
    state_path = debate_dir / "state.json"
    if not state_path.exists():
        console.print(f"[red]Debate '{debate_id}' not found in {DEBATES_DIR}[/red]")
        raise typer.Exit(1)

    state = _load_state_or_exit(state_path, debate_id)
    if state.is_finished:
        console.print(f"[yellow]Debate already finished ({state.status}).[/yellow]")
        _finish_banner(debate_dir, state.status)
        return

    console.print(Panel.fit(
        f"[bold]Resuming[/bold] {state.debate_id}\n"
        f"round {state.current_round}, phase {state.current_phase.value}\n"
        f"[dim]{state.prompt[:120]}[/dim]",
        border_style="blue",
    ))

    try:
        status = asyncio.run(run_debate(state, debate_dir, stall_timeout))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130) from None

    _finish_banner(debate_dir, status)


@app.command("list")
def list_debates() -> None:
    """List all debates."""
    if not DEBATES_DIR.exists() or not any(DEBATES_DIR.iterdir()):
        console.print("[dim]No debates yet.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Round")
    table.add_column("Prompt")
    for d in sorted(DEBATES_DIR.iterdir(), reverse=True):
        state_path = d / "state.json"
        if not state_path.exists():
            table.add_row(d.name, "[dim]no state[/dim]", "-", "")
            continue
        try:
            s = load_state(state_path)
            color = {"done": "green", "deadlocked": "yellow"}.get(s.status, "cyan")
            table.add_row(
                s.debate_id, f"[{color}]{s.status}[/{color}]",
                str(s.current_round), s.prompt[:60].replace("\n", " "),
            )
        except Exception:
            table.add_row(d.name, "[red]error[/red]", "-", "")
    console.print(table)


@app.command()
def status(debate_id: str = typer.Argument(..., help="Debate ID")) -> None:
    """Show current round/phase and who has contributed."""
    debate_dir = _resolve_debate_dir(debate_id)
    state_path = debate_dir / "state.json"
    if not state_path.exists():
        console.print(f"[red]Debate '{debate_id}' not found.[/red]")
        raise typer.Exit(1)

    state = _load_state_or_exit(state_path, debate_id)
    console.print(Panel.fit(
        f"[bold]{state.debate_id}[/bold]  status=[bold]{state.status}[/bold]  "
        f"round {state.current_round}/{state.max_rounds}  phase={state.current_phase.value}\n"
        f"[dim]{state.prompt[:120]}[/dim]",
        border_style="blue",
    ))
    console.print(_progress_table(state))
    footer = _phase_footer(state)
    if footer:
        console.print(footer)


@app.command()
def show(debate_id: str = typer.Argument(..., help="Debate ID to show")) -> None:
    """Render the final consensus document for a debate."""
    debate_dir = _resolve_debate_dir(debate_id)
    final_file = debate_dir / "final.md"
    state_path = debate_dir / "state.json"
    if not state_path.exists():
        console.print(f"[red]Debate '{debate_id}' not found.[/red]")
        raise typer.Exit(1)
    if final_file.exists():
        console.print(Markdown(final_file.read_text()))
    else:
        s = _load_state_or_exit(state_path, debate_id)
        console.print(f"[yellow]No final answer yet[/yellow] (status: {s.status}, "
                      f"round {s.current_round}/{s.max_rounds}, phase {s.current_phase.value}).")


if __name__ == "__main__":
    app()
