"""Terminal theme and Rich console configuration using Strataforge brand tokens."""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.theme import Theme

# Reconfigure Windows stdout/stderr to UTF-8 if supported
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Strataforge Brand Color Palette
COLOR_BG = "#17151F"
COLOR_PLUM = "#272039"
COLOR_COPPER = "#E9AC74"
COLOR_SAGE = "#91C8A8"
COLOR_CORAL = "#E77770"
COLOR_GOLD = "#ECC06C"
COLOR_LAVENDER = "#B892FF"
COLOR_MUTED = "#8E83A3"
COLOR_TEXT = "#F2EFF7"

STRATAFORGE_THEME = Theme(
    {
        "copper": f"bold {COLOR_COPPER}",
        "copper.dim": COLOR_COPPER,
        "sage": f"bold {COLOR_SAGE}",
        "coral": f"bold {COLOR_CORAL}",
        "gold": f"bold {COLOR_GOLD}",
        "lavender": COLOR_LAVENDER,
        "plum": COLOR_PLUM,
        "muted": COLOR_MUTED,
        "text": COLOR_TEXT,
        "badge.pass": f"bold {COLOR_SAGE} on #1E332A",
        "badge.info": f"bold {COLOR_LAVENDER} on #2B2342",
        "badge.warn": f"bold {COLOR_GOLD} on #362E1A",
        "badge.crit": f"bold {COLOR_CORAL} on #3B1C1C",
        "badge.type": f"{COLOR_MUTED} on #272039",
    }
)


def create_console(no_color: bool = False) -> Console:
    """Create a configured Rich Console respecting --no-color flag and NO_COLOR env var."""
    force_no_color = no_color or bool(os.getenv("NO_COLOR"))
    return Console(
        theme=STRATAFORGE_THEME,
        color_system=None if force_no_color else "auto",
        no_color=force_no_color,
        highlight=not force_no_color,
        safe_box=True,
    )
