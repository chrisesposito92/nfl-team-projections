# src/pbp/anchor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .priors import GamePriors, TeamPriors


@dataclass
class AnchorWeights:
    """Single knob for now. 0 = engine only, 1 = match target exactly (clamped)."""
    yards_weight: float = 0.35
    clamp_min: float = 0.60
    clamp_max: float = 1.40


def _team_from_cols(df: pd.DataFrame) -> Tuple[str, str]:
    # Assumes 'home'/'away' columns are present (your engine already writes them).
    h = df["home"].dropna().iloc[0]
    a = df["away"].dropna().iloc[0]
    return str(h), str(a)


def _pcomp(tp: TeamPriors) -> float:
    # Priors comp% (comp_exp + cpoe if available), safely clamped.
    p = float(tp.comp_exp)
    try:
        p += float(getattr(tp, "cpoe", 0.0))
    except Exception:
        pass
    return float(np.clip(p, 0.45, 0.75))


def _estimate_targets_from_priors(df: pd.DataFrame, tp: TeamPriors) -> Dict[str, float]:
    """
    VERY light estimator used only to set yard anchors for a single simulated game:
      attempts ~ offensive plays * pass_rate
      completions ~ attempts * pcomp
      pass yards ~ completions * (aDOT + yac_mean)
      rush yards ~ rush attempts * rush_ypc
    """
    n_off = int((df["offense"] == tp.team).sum())
    att = n_off * float(np.clip(tp.pass_rate, 0.35, 0.75))
    comp = att * _pcomp(tp)
    pass_yards = comp * max(2.0, tp.aDOT) + comp * max(0.5, tp.yac_mean)
    rush_att = n_off - att
    rush_yards = rush_att * float(np.clip(tp.rush_ypc, 3.0, 6.5))
    return {"pass_yards": float(pass_yards), "rush_yards": float(rush_yards)}


def _current_team_yards(df: pd.DataFrame, team: str) -> Dict[str, float]:
    # Pass yards: exclude sacks & incompletions; keep only completed pass yardage.
    mask_team = (df["offense"] == team)
    pass_mask = mask_team & (df["play_type"] == "pass") & (df["is_complete"] == True) & (df["is_sack"] == False)
    run_mask = mask_team & (df["play_type"] == "run")
    pass_yards = float(df.loc[pass_mask, "yards_gained"].sum())
    rush_yards = float(df.loc[run_mask, "yards_gained"].sum())
    return {"pass_yards": pass_yards, "rush_yards": rush_yards}


def soft_anchor_yards(df: pd.DataFrame, priors: GamePriors, weights: AnchorWeights | None = None) -> pd.DataFrame:
    """
    Post-process a single simulated game's PBP rows to nudge pass/rush yards
    toward simple targets implied by priors. PURELY cosmetic (does not resim engine).
    - Scales per-play yards for completed passes and rushes for each offense.
    - Skips sacks and TD plays to avoid breaking bookkeeping.
    """
    if weights is None or weights.yards_weight <= 0.0:
        return df

    w = float(np.clip(weights.yards_weight, 0.0, 1.0))
    cmin = float(weights.clamp_min)
    cmax = float(weights.clamp_max)

    home, away = _team_from_cols(df)

    def _scale_side(team: str, tp: TeamPriors) -> pd.Series:
        cur = _current_team_yards(df, team)
        tgt = _estimate_targets_from_priors(df, tp)
        scale_pass = (1 - w) + w * ((tgt["pass_yards"] + 1e-9) / max(cur["pass_yards"], 1e-9))
        scale_rush = (1 - w) + w * ((tgt["rush_yards"] + 1e-9) / max(cur["rush_yards"], 1e-9))
        return pd.Series({
            "scale_pass": float(np.clip(scale_pass, cmin, cmax)),
            "scale_rush": float(np.clip(scale_rush, cmin, cmax)),
        })

    s_home = _scale_side(home, priors.home)
    s_away = _scale_side(away, priors.away)

    out = df.copy()

    # Scale completed pass yards (exclude sacks and TDs to avoid contradictions)
    mask_home_pass = (out["offense"] == home) & (out["play_type"] == "pass") & (out["is_complete"] == True) & (out["is_sack"] == False) & (out["result"] != "TD")
    mask_away_pass = (out["offense"] == away) & (out["play_type"] == "pass") & (out["is_complete"] == True) & (out["is_sack"] == False) & (out["result"] != "TD")
    out.loc[mask_home_pass, "yards_gained"] = out.loc[mask_home_pass, "yards_gained"] * s_home["scale_pass"]
    out.loc[mask_away_pass, "yards_gained"] = out.loc[mask_away_pass, "yards_gained"] * s_away["scale_pass"]

    # Scale rush yards (skip TDs to avoid contradictions)
    mask_home_run = (out["offense"] == home) & (out["play_type"] == "run") & (out["result"] != "TD")
    mask_away_run = (out["offense"] == away) & (out["play_type"] == "run") & (out["result"] != "TD")
    out.loc[mask_home_run, "yards_gained"] = out.loc[mask_home_run, "yards_gained"] * s_home["scale_rush"]
    out.loc[mask_away_run, "yards_gained"] = out.loc[mask_away_run, "yards_gained"] * s_away["scale_rush"]

    # Optional: round to 3 decimals for nicer CSVs
    out["yards_gained"] = out["yards_gained"].astype(float).round(3)
    return out