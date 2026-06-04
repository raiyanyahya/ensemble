from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Route ensemble logs through rich. Idempotent."""
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger("ensemble")
    root.setLevel(level)
    if root.handlers:
        for h in root.handlers:
            h.setLevel(level)
        return
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    handler.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


# Per-model accent colors for consistent styling across the CLI.
MODEL_COLORS = {
    "gpt4o": "green",
    "claude": "magenta",
    "deepseek": "cyan",
}


def model_style(name: str) -> str:
    return MODEL_COLORS.get(name, "white")
