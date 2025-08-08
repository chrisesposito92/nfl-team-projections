from __future__ import annotations
from functools import lru_cache
import numpy as np
from .data import load_defense_profile
from .config import (
    ADJUSTMENTS_ENABLED,
    DEF_LOOKBACK_SEASONS,
    DEF_PASS_YDS_W, DEF_PASS_ATT_W, DEF_PASS_TDS_W,
    DEF_RUSH_YDS_W, DEF_RUSH_ATT_W, DEF_RUSH_TDS_W,
    HOME_BONUS_PASS_YDS, HOME_BONUS_PASS_ATT,
    HOME_BONUS_RUSH_YDS, HOME_BONUS_RUSH_ATT,
)

@lru_cache(maxsize=8)
def _def_profile_cached(year: int):
    return load_defense_profile(year, DEF_LOOKBACK_SEASONS)

def adjust_team_totals(base: dict, opponent: str, is_home: bool, year: int) -> dict:
    """
    Adjusts baseline team totals given opponent defensive strength and home/away.

    Args:
        base: dict with keys: pass_attempts, pass_yards, pass_tds, rush_attempts, rush_yards, rush_tds
        opponent: opponent team abbreviation (e.g., 'CLE')
        is_home: True if home game
        year: projection season (e.g., 2025)

    Returns:
        dict with adjusted team totals (non-negative)
    """
    if not ADJUSTMENTS_ENABLED:
        return base

    prof = _def_profile_cached(year)
    if opponent not in prof.index:
        return base

    row = prof.loc[opponent]
    z_pass = float(row["pass_epa_allowed_z"])
    z_rush = float(row["rush_epa_allowed_z"])
    z_neutral_pr = float(row["neutral_pass_rate_allowed_z"])

    out = dict(base)

    out["pass_yards"] = max(
        0.0,
        out["pass_yards"] * (1.0 + DEF_PASS_YDS_W * z_pass + (HOME_BONUS_PASS_YDS if is_home else 0.0)),
    )
    out["pass_attempts"] = max(
        0.0,
        out["pass_attempts"] * (1.0 + DEF_PASS_ATT_W * z_neutral_pr + (HOME_BONUS_PASS_ATT if is_home else 0.0)),
    )
    out["pass_tds"] = max(
        0.0,
        out["pass_tds"] * (1.0 + DEF_PASS_TDS_W * z_pass),
    )

    out["rush_yards"] = max(
        0.0,
        out["rush_yards"] * (1.0 + DEF_RUSH_YDS_W * z_rush + (HOME_BONUS_RUSH_YDS if is_home else 0.0)),
    )
    out["rush_attempts"] = max(
        0.0,
        out["rush_attempts"] * (1.0 + DEF_RUSH_ATT_W * (-z_neutral_pr) + (HOME_BONUS_RUSH_ATT if is_home else 0.0)),
    )
    out["rush_tds"] = max(
        0.0,
        out["rush_tds"] * (1.0 + DEF_RUSH_TDS_W * z_rush),
    )

    return out