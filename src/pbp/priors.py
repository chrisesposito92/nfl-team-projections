from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from .config import (
    BASE_PASS_RATE, SACK_RATE_BASE, INT_RATE_BASE, SCREEN_PROB_BASE,
    RUN_LN_SIGMA
)

__all__ = ["TeamPriors", "GamePriors", "get_team_priors", "build_game_priors"]

@dataclass
class TeamPriors:
    team: str
    # Offense tendencies / efficiencies
    pass_rate: float = BASE_PASS_RATE
    aDOT: float = 8.5
    comp_exp: float = 0.63          # expected completion fraction
    cpoe: float = 0.0               # completion over expectation
    yac_mean: float = 4.6           # avg YAC per completion
    yac_aoe: float = 0.0            # YAC above expectation
    sack_rate: float = SACK_RATE_BASE
    int_rate: float = INT_RATE_BASE
    screen_rate: float = SCREEN_PROB_BASE
    rush_ypc: float = 4.3
    rush_roe_pa: float = 0.0        # rush yards over expected per attempt
    box_rate_8: float = 0.20        # share of carries w/ 8+ in box
    tempo_sec: float = 28.0
    hurry_sec: float = 18.0
    # Simple defense modifiers (Phase-1: light touch; you can elaborate later)
    pass_def_factor: float = 0.0    # +/- adjusts comp_exp and aDOT
    rush_def_factor: float = 0.0    # +/- adjusts YPC

@dataclass
class GamePriors:
    home: TeamPriors
    away: TeamPriors

# ---------- Builders (use your loaders if available, else defaults) ----------

def _safe_weighted_mean(x: pd.Series, w: pd.Series, default: float) -> float:
    try:
        w = w.fillna(0.0)
        x = x.fillna(0.0)
        s = float(w.sum())
        return float((x * w).sum() / s) if s > 0 else default
    except Exception:
        return default

def _ngs_team_offense(year: int) -> pd.DataFrame:
    """
    Optional helper if you expose import_ngs_data(kind, years).
    Expected to return a frame keyed by 'team_abbr' with offense aggregates.
    """
    try:
        from ..data import import_ngs_data  # your existing loader, if present
    except Exception:
        return pd.DataFrame()

    # Receiving (YAC, AOE)
    rec = import_ngs_data("receiving", [year])
    # Passing (CPOE, intended air)
    pas = import_ngs_data("passing", [year])
    # Rushing (ROE)
    rus = import_ngs_data("rushing", [year])

    # Aggregate safely
    rec_g = rec.groupby("team_abbr", as_index=False).agg(
        avg_yac=("avg_yac", "mean"),
        avg_yac_aoe=("avg_yac_above_expectation", "mean"),
        avg_sep=("avg_separation", "mean"),
    )

    def _wmean(df, val, w):
        return df.groupby("team_abbr").apply(
            lambda d: _safe_weighted_mean(d[val], d[w], float(d[val].mean()))
        ).reset_index(name=f"w_{val}")

    pas_w_cpoe = _wmean(pas, "completion_percentage_above_expectation", "attempts")
    pas_w_xcp = _wmean(pas, "expected_completion_percentage", "attempts")
    pas_w_iair = _wmean(pas, "avg_intended_air_yards", "attempts")

    rus_g = rus.groupby("team_abbr", as_index=False).agg(
        avg_rush_yds=("avg_rush_yards", "mean"),
        roe_pa=("rush_yards_over_expected_per_att", "mean"),
        box8=("percent_attempts_gte_eight_defenders", "mean"),
    )

    out = (
        rec_g.merge(pas_w_cpoe, on="team_abbr", how="outer")
             .merge(pas_w_xcp, on="team_abbr", how="outer")
             .merge(pas_w_iair, on="team_abbr", how="outer")
             .merge(rus_g, on="team_abbr", how="outer")
    )
    return out.rename(columns={"team_abbr": "team"})


def _ftn_screen_blitz(year: int) -> pd.DataFrame:
    try:
        from ..data import import_ftn_data
    except Exception:
        return pd.DataFrame()

    ftn = import_ftn_data([year])
    # Simple team-level screen and blitz proxies
    g = ftn.groupby("nflverse_game_id").agg(
        screen_rate=("is_screen_pass", "mean"),
        blitzers=("n_blitzers", "mean"),
        pass_rushers=("n_pass_rushers", "mean"),
    ).reset_index()

    # Extract home/away abbr from nflverse_game_id if present (e.g., "2024_01_BAL_KC")
    # MVP: average by either team tag found in id string
    def teams_from_id(s: str) -> list[str]:
        parts = str(s).split("_")
        if len(parts) >= 4:
            return [parts[-2], parts[-1]]
        return []

    rows = []
    for r in g.itertuples(index=False):
        teams = teams_from_id(r.nflverse_game_id)
        for t in teams:
            rows.append({"team": t, "screen_rate": r.screen_rate, "blitzers": r.blitzers})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("team", as_index=False).mean()

def get_team_priors(team: str, year: int) -> TeamPriors:
    tp = TeamPriors(team=team)
    try:
        ngs = _ngs_team_offense(year)
        if not ngs.empty and team in set(ngs["team"]):
            row = ngs.loc[ngs["team"] == team].iloc[0]
            tp.aDOT = float(row.get("w_avg_intended_air_yards", tp.aDOT))
            tp.comp_exp = float(row.get("w_expected_completion_percentage", tp.comp_exp)) / 100.0
            tp.cpoe = float(row.get("w_completion_percentage_above_expectation", 0.0)) / 100.0
            tp.yac_mean = float(row.get("avg_yac", tp.yac_mean))
            tp.yac_aoe = float(row.get("avg_yac_aoe", 0.0))
            tp.rush_ypc = float(row.get("avg_rush_yds", tp.rush_ypc))
            tp.rush_roe_pa = float(row.get("roe_pa", tp.rush_roe_pa))
            tp.box_rate_8 = float(row.get("box8", tp.box_rate_8)) / 100.0
    except Exception:
        pass

    try:
        ftn = _ftn_screen_blitz(year)
        if not ftn.empty and team in set(ftn["team"]):
            row = ftn.loc[ftn["team"] == team].iloc[0]
            tp.screen_rate = float(row.get("screen_rate", tp.screen_rate))
    except Exception:
        pass

    # Clamp & sanity (no CPOE baked into comp_exp here)
    tp.pass_rate = float(np.clip(tp.pass_rate, 0.40, 0.70))
    tp.comp_exp = float(np.clip(tp.comp_exp, 0.50, 0.72))
    tp.yac_mean = max(0.5, tp.yac_mean + tp.yac_aoe)
    tp.rush_ypc = float(np.clip(tp.rush_ypc + tp.rush_roe_pa, 3.5, 5.6))
    tp.screen_rate = float(np.clip(tp.screen_rate, 0.03, 0.25))
    return tp

def build_game_priors(home: str, away: str, year: int) -> GamePriors:
    """Convenience wrapper to build both teams' priors."""
    return GamePriors(
        home=get_team_priors(home, year),
        away=get_team_priors(away, year),
    )