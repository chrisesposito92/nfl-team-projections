# src/pbp_calibrate.py
from __future__ import annotations
import os
import sys
import argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

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


def main():
    p = argparse.ArgumentParser("PBP Calibration Harness")
    # Schedule sources
    p.add_argument("--schedule-csv", type=str, default="", help="Optional prebuilt schedule CSV.")
    p.add_argument("--years", type=int, nargs="+", default=[], help="Seasons to load via nfl_data_py.import_schedules")
    p.add_argument("--season-type", type=str, default="REG", help="REG or POST (default REG)")

    # Optional single-game filter on top of schedule
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--home", type=str, default=None)
    p.add_argument("--away", type=str, default=None)
    p.add_argument("--week", type=int, default=None)

    # Sim params
    p.add_argument("--sims-per-game", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Limit number of games (debug/smoke)")

    # Engine toggles (parsed for bookkeeping; simulate_game uses your defaults)
    p.add_argument("--engine", choices=["stateful","simple"], default="stateful")
    p.add_argument("--penalties", choices=["on","off"], default="off")
    p.add_argument("--anchor", choices=["on","off"], default="off")

    # Outputs
    p.add_argument("--out", type=str, default="artifacts/pbp_cal", help="Output directory")

    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    sched = _load_schedule(args)
    sched = _filter_schedule(sched, args)

    if sched.empty:
        print("No games after filtering. Check your flags.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(sched)} games | sims/game={args.sims_per_game} | engine={args.engine} | penalties={args.penalties} | anchor={args.anchor}")

    summary_rows: List[Dict[str, Any]] = []
    pit_rows: List[Dict[str, Any]] = []

    base_seed = int(args.seed)

    for i, r in enumerate(sched.itertuples(index=False), start=1):
        season = int(getattr(r, "season"))
        week = int(getattr(r, "week"))
        home = str(getattr(r, "home"))
        away = str(getattr(r, "away"))

        seed_i = base_seed + i  # simple per-game jitter

        h_draws, a_draws = _sim_points_for_game(home, away, season, args.sims_per_game, seed_i)
        sim_ok = (h_draws.size == args.sims_per_game and a_draws.size == args.sims_per_game)

        # summaries
        hs = _summarize(h_draws)
        asu = _summarize(a_draws)
        p_home_win = float(np.mean(h_draws > a_draws)) if h_draws.size and a_draws.size else np.nan

        # truth (if available on schedule)
        home_truth = float(getattr(r, "home_score_truth", np.nan)) if hasattr(r, "home_score_truth") else np.nan
        away_truth = float(getattr(r, "away_score_truth", np.nan)) if hasattr(r, "away_score_truth") else np.nan
        home_pit = _pit_value(home_truth, h_draws)
        away_pit = _pit_value(away_truth, a_draws)

        summary_rows.append({
            "season": season,
            "week": week,
            "home": home,
            "away": away,
            "engine": args.engine,
            "penalties": args.penalties,
            "anchor": args.anchor,
            "sims": int(args.sims_per_game),
            "ok": bool(sim_ok),
            # predictions
            "pred_home_mean": hs["mean"],
            "pred_home_p10":  hs["p10"],
            "pred_home_p50":  hs["p50"],
            "pred_home_p90":  hs["p90"],
            "pred_away_mean": asu["mean"],
            "pred_away_p10":  asu["p10"],
            "pred_away_p50":  asu["p50"],
            "pred_away_p90":  asu["p90"],
            "pred_p_home_win": p_home_win,
            # truth (if present)
            "truth_home": home_truth,
            "truth_away": away_truth,
            # PIT (for later histograms)
            "pit_home_points": home_pit,
            "pit_away_points": away_pit,
        })

        pit_rows.append({
            "season": season,
            "week": week,
            "home": home,
            "away": away,
            "team": home,
            "pit_points": home_pit,
        })
        pit_rows.append({
            "season": season,
            "week": week,
            "home": home,
            "away": away,
            "team": away,
            "pit_points": away_pit,
        })

        if (i % 25) == 0 or i == len(sched):
            print(f"  {i:>4}/{len(sched)}  {away}@{home}  mean {asu['mean']:.1f}-{hs['mean']:.1f}  p(H win)={p_home_win:.3f}")

    # Write artifacts
    summary_df = pd.DataFrame(summary_rows)
    pit_df = pd.DataFrame(pit_rows)

    sum_path = os.path.join(args.out, "summary_points.csv")
    pit_path = os.path.join(args.out, "pit_points.csv")
    summary_df.to_csv(sum_path, index=False)
    pit_df.to_csv(pit_path, index=False)

    print("\nWrote:")
    print(" ", sum_path)
    print(" ", pit_path)
    # Quick overall sanity print
    if "truth_home" in summary_df.columns and summary_df["truth_home"].notna().any():
        valid = summary_df.dropna(subset=["truth_home","truth_away"])
        mae = np.abs((valid["pred_home_mean"] - valid["truth_home"])) \
              + np.abs((valid["pred_away_mean"] - valid["truth_away"]))
        mae = float(mae.mean()) if len(valid) else np.nan
        print(f"Mean absolute error (sum of teams) ~ {mae:.2f} points over {len(valid)} games.")
    else:
        print("No truth in schedule; run produced predictions only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())