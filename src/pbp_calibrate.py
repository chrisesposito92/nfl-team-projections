# src/pbp_calibrate.py
from __future__ import annotations
import os
import sys
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from nfl_data_py import import_schedules
from .pbp.calibrate import run_calibration

# Engine & priors (your package)
from .pbp import simulate_game, build_game_priors

# Optional: schedules from nfl_data_py (we'll fall back gracefully)
def _try_import_schedules(years: List[int]) -> pd.DataFrame:
    try:
        from nfl_data_py import import_schedules
    except Exception:
        return pd.DataFrame()
    try:
        df = import_schedules(years)
        # Normalize columns we need
        cols = {c.lower(): c for c in df.columns}
        # Expected: season, week, home_team, away_team, home_score, away_score, season_type
        need = ["season", "week", "home_team", "away_team"]
        for n in need:
            if n not in cols:
                # If any required column isn't there, signal failure
                return pd.DataFrame()
        # Create a normalized output
        out = pd.DataFrame({
            "season": df[cols["season"]],
            "week": df[cols["week"]],
            "home": df[cols["home_team"]].astype(str).str.upper(),
            "away": df[cols["away_team"]].astype(str).str.upper()
        })
        # Keep truth if available
        if "home_score" in cols and "away_score" in cols:
            out["home_score_truth"] = pd.to_numeric(df[cols["home_score"]], errors="coerce")
            out["away_score_truth"] = pd.to_numeric(df[cols["away_score"]], errors="coerce")
        else:
            out["home_score_truth"] = np.nan
            out["away_score_truth"] = np.nan

        # Filter regular season by default; caller may override
        if "season_type" in cols:
            st = df[cols["season_type"]].astype(str).str.upper()
            out["season_type"] = st
        else:
            out["season_type"] = "REG"

        # Keep only numeric weeks (drop e.g., SB named rows)
        out = out[pd.to_numeric(out["week"], errors="coerce").notna()].copy()
        out["week"] = out["week"].astype(int)
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _load_schedule(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load a schedule either from --schedule-csv or using nfl_data_py.import_schedules via --years.
    """
    if args.schedule_csv:
        df = pd.read_csv(args.schedule_csv)
        # Normalize columns to: season, week, home, away, [home_score_truth, away_score_truth, season_type]
        lower = {c.lower(): c for c in df.columns}
        def col(*names, optional=False, default=np.nan):
            for n in names:
                if n in lower:
                    return df[lower[n]]
            if optional:
                return pd.Series([default] * len(df))
            raise ValueError(f"Missing required column among {names} in {args.schedule_csv}")

        out = pd.DataFrame({
            "season": pd.to_numeric(col("season").astype(str), errors="coerce").astype(int),
            "week": pd.to_numeric(col("week").astype(str), errors="coerce").astype(int),
            "home": col("home","home_team").astype(str).str.upper(),
            "away": col("away","away_team").astype(str).str.upper(),
            "home_score_truth": pd.to_numeric(col("home_score","home_points", optional=True), errors="coerce"),
            "away_score_truth": pd.to_numeric(col("away_score","away_points", optional=True), errors="coerce"),
            "season_type": col("season_type", optional=True, default="REG").astype(str).str.upper(),
        })
        return out.reset_index(drop=True)

    if not args.years:
        raise SystemExit("Provide either --schedule-csv or --years ...")

    df = _try_import_schedules(args.years)
    if df.empty:
        raise SystemExit(
            "Could not load schedules via nfl_data_py.import_schedules. "
            "Install nfl_data_py or provide --schedule-csv."
        )
    return df.reset_index(drop=True)


def _filter_schedule(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df2 = df.copy()
    # Season type filter
    if args.season_type:
        df2 = df2[df2["season_type"].str.upper() == args.season_type.upper()]
    # Optional specific year/week/home/away selection (useful for spot checks)
    if args.year:
        df2 = df2[df2["season"] == int(args.year)]
    if args.week:
        df2 = df2[df2["week"] == int(args.week)]
    if args.home:
        df2 = df2[df2["home"].str.upper() == args.home.upper()]
    if args.away:
        df2 = df2[df2["away"].str.upper() == args.away.upper()]
    # Optionally limit number of games
    if args.limit and len(df2) > args.limit:
        df2 = df2.head(args.limit)
    return df2.reset_index(drop=True)


def _sim_points_for_game(home: str, away: str, year: int, sims: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the engine and return arrays of simulated points for home & away.
    """
    priors = build_game_priors(home, away, year)
    # Simulate
    res = simulate_game(home, away, year, n_sims=sims, seed=seed, priors=priors)
    # Extract final scores from each sim df (last row = END)
    h_pts = []
    a_pts = []
    for df in res:
        if df.empty:
            continue
        final = df.iloc[-1]
        h = float(final.get("home_score", np.nan))
        a = float(final.get("away_score", np.nan))
        # Some engines store them as NaN on penultimate line — but last "END" line should be set.
        h_pts.append(h)
        a_pts.append(a)
    return np.array(h_pts, dtype=float), np.array(a_pts, dtype=float)


def _summarize(points: np.ndarray) -> Dict[str, float]:
    if points.size == 0:
        return {"mean": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}
    return {
        "mean": float(np.mean(points)),
        "p10": float(np.percentile(points, 10)),
        "p50": float(np.percentile(points, 50)),
        "p90": float(np.percentile(points, 90)),
    }


def _pit_value(truth: float, draws: np.ndarray) -> float:
    """
    PIT for a continuous outcome = empirical CDF at truth.
    """
    if not np.isfinite(truth) or draws.size == 0:
        return np.nan
    return float(np.mean(draws <= truth))

def build_schedule(years, season_type="REG", weeks=None) -> pd.DataFrame:
    df = import_schedules(years)
    # nfl_data_py uses "season_type" or "game_type" depending on version
    if "season_type" in df.columns:
        df = df[df["season_type"] == season_type]
    elif "game_type" in df.columns:
        df = df[df["game_type"] == season_type]
    cols = {}
    if "home_team" in df.columns: cols["home"] = "home_team"
    elif "home" in df.columns:   cols["home"] = "home"
    if "away_team" in df.columns: cols["away"] = "away_team"
    elif "away" in df.columns:    cols["away"] = "away"
    cols["season"] = "season"
    cols["week"] = "week"
    df = df[list(cols.values())].rename(columns={v:k for k,v in cols.items()})
    df["home"] = df["home"].str.upper()
    df["away"] = df["away"].str.upper()
    if weeks:
        df = df[df["week"].isin(weeks)]
    return df[["season","week","home","away"]].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser("PBP Calibration Harness")
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--season-type", choices=["REG","POST"], default="REG")
    p.add_argument("--weeks", nargs="*", type=int, default=None)
    p.add_argument("--sims-per-game", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--engine", choices=["stateful","simple"], default="stateful")
    p.add_argument("--penalties", choices=["on","off"], default="off")
    p.add_argument("--anchor", choices=["on","off"], default="off")
    p.add_argument("--truth-csv", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    args = p.parse_args()

    schedule = build_schedule(args.years, season_type=args.season_type, weeks=args.weeks)
    out = run_calibration(
        schedule,
        sims_per_game=args.sims_per_game,
        seed=args.seed,
        engine=args.engine,
        penalties=(args.penalties == "on"),
        anchor=(args.anchor == "on"),
        out_dir=args.outdir,
        truth_csv=args.truth_csv,
    )

    print("Calibration complete.")
    if args.outdir:
        print(f"Artifacts written to: {args.outdir}")

if __name__ == "__main__":
    sys.exit(main())