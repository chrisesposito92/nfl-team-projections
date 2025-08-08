from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List
from .config import SIM_CALIBRATION_FILE
from .data import load_pbp
from .features import build_team_game_stats_from_pbp

COLS_A = ["pass_attempts", "rush_attempts", "pass_yards", "rush_yards"]
COLS_TD = ["pass_tds", "rush_tds"]

def _neg_binom_theta(series: pd.Series) -> float:
    s = series.dropna().astype(float)
    if s.empty:
        return 0.0
    m = s.mean()
    v = s.var(ddof=1)
    if v <= m or m <= 0:
        return 0.0
    return float((m * m) / (v - m))

def build_calibration(years: List[int]) -> Dict[str, object]:
    pbp = load_pbp(years)
    team_games = build_team_game_stats_from_pbp(pbp)
    team_games = team_games[COLS_A + COLS_TD].dropna().reset_index(drop=True)

    sigma_frac = {}
    for c in COLS_A:
        mean_c = team_games[c].mean()
        std_c = team_games[c].std(ddof=1)
        sf = float(std_c / max(mean_c, 1.0))
        sigma_frac[c] = sf

    A = team_games[COLS_A].astype(float).to_numpy()
    rho = np.corrcoef(A, rowvar=False)
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

    nb_theta = {c: _neg_binom_theta(team_games[c]) for c in COLS_TD}

    return {
        "sigma_frac": sigma_frac,
        "rho": rho.tolist(),
        "nb_theta": nb_theta,
    }

def main():
    years = list(range(2011, 2025))
    calib = build_calibration(years)
    os.makedirs(os.path.dirname(SIM_CALIBRATION_FILE), exist_ok=True)
    with open(SIM_CALIBRATION_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved calibration to {SIM_CALIBRATION_FILE}")

if __name__ == "__main__":
    main()