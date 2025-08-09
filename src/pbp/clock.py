from __future__ import annotations
import math
from .config import (
    BASE_TEMPO_SEC, HURRY_TEMPO_SEC, NOHUDDLE_TEMPO_SEC,
    INCOMPLETE_PLAY_TIME, QTR_SECONDS
)
from .state import GameState

def burn_clock_seconds(state: GameState, base_seconds: float, incomplete: bool) -> None:
    """
    MVP: If incomplete => burn only play time (no runoff). Else burn base tempo seconds.
    Late-game hurry-up reduces time between snaps.
    """
    if incomplete:
        dt = INCOMPLETE_PLAY_TIME
    else:
        # crude hurry-up: last 2 minutes of 2nd/4th quarters
        hurry = (state.quarter in (2, 4)) and (state.clock <= 120)
        dt = HURRY_TEMPO_SEC if hurry else base_seconds

    # Apply to current quarter; if it rolls past, let engine handle quarter transitions
    state.clock = max(0, int(state.clock - round(dt)))

def next_quarter(state: GameState) -> None:
    state.quarter += 1
    state.clock = QTR_SECONDS

def yardline_after_gain(yardline: int, gain: float) -> int:
    ny = int(max(1, min(99, round(yardline + gain))))
    return ny