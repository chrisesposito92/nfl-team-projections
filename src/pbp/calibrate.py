# src/pbp/calibrate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict, List
import numpy as np
import pandas as pd

# We depend only on the public pbp surface you already expose
from . import (
    simulate_game as simulate_simple,
    simulate_single_game as simulate_stateful,
    build_game_priors,
)

# --------- What we calibrate by default ----------
CALIB_METRICS: Tuple[str, ...] = (
    "points",
    "plays",
    "pass_att",
    "completions",
    "sacks",
    "net_pass_yards",   # pass yards incl. sack yardage (negative)
    "rush_att",
    "rush_yards",
    "pass_tds",
    "rush_tds",
)

DEFAULT_QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class GameKey:
    season: int
    week: int
    home: str
    away: str


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def team_box_from_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """
    Return two rows (home, away) with team game totals that are stable for calibration.
    Expects simulator columns:
      offense, defense, play_type, yards_gained, points, is_complete, is_sack,
      result, drive_id, home, away, home_score, away_score
    """
    if plays.empty:
        return pd.DataFrame(columns=["team","opp","points","plays","pass_att","completions",
                                     "sacks","net_pass_yards","rush_att","rush_yards",
                                     "pass_tds","rush_tds","season","week","home","away"])

    _ensure_cols(plays, ["is_complete", "is_sack", "result", "home", "away"])
    # Use END row for final scores if present
    end = plays.iloc[-1] if str(plays.iloc[-1].get("result")) == "END" else None

    home = str(plays["home"].iloc[0])
    away = str(plays["away"].iloc[0])

    # Helper to aggregate per offense team
    def agg_for(team: str) -> Dict[str, float]:
        sub = plays[plays["offense"] == team].copy()

        pass_mask = (sub["play_type"] == "pass")
        run_mask  = (sub["play_type"] == "run")
        sack_mask = pass_mask & sub["is_sack"].fillna(False)

        # Attempts exclude sacks
        pass_att = int((pass_mask & ~sack_mask).sum())
        completions = int((pass_mask & sub["is_complete"].fillna(False)).sum())
        sacks = int(sack_mask.sum())

        pass_yards = float(sub.loc[pass_mask, "yards_gained"].fillna(0.0).sum())  # includes sacks (neg)
        rush_att   = int(run_mask.sum())
        rush_yards = float(sub.loc[run_mask, "yards_gained"].fillna(0.0).sum())

        pass_tds = int((pass_mask & (sub["result"] == "TD")).sum())
        rush_tds = int((run_mask  & (sub["result"] == "TD")).sum())

        scrimmage_plays = int((pass_mask | run_mask).sum())

        return dict(
            plays=scrimmage_plays,
            pass_att=pass_att,
            completions=completions,
            sacks=sacks,
            net_pass_yards=pass_yards,
            rush_att=rush_att,
            rush_yards=rush_yards,
            pass_tds=pass_tds,
            rush_tds=rush_tds,
        )

    home_row = agg_for(home)
    away_row = agg_for(away)

    # Score from END row if present (preferred)
    if end is not None and "home_score" in end and "away_score" in end:
        home_pts = float(end["home_score"])
        away_pts = float(end["away_score"])
    else:
        # Fallback: sum the points the offense scored (rarely misses defensive scores, but OK for MVP)
        home_pts = float(plays.loc[plays["offense"] == home, "points"].fillna(0.0).sum())
        away_pts = float(plays.loc[plays["offense"] == away, "points"].fillna(0.0).sum())

    # Attach points + keys
    for row, team, opp, pts in ((home_row, home, away, home_pts), (away_row, away, home, away_pts)):
        row["team"] = team
        row["opp"] = opp
        row["points"] = pts
        row["home"] = home
        row["away"] = away

    out = pd.DataFrame([home_row, away_row])
    # Try to pick season/week if present
    for c in ("season", "week"):
        if c in plays.columns:
            out[c] = plays[c].iloc[0]
        else:
            out[c] = np.nan
    return out[["team","opp","points","plays","pass_att","completions","sacks",
                "net_pass_yards","rush_att","rush_yards","pass_tds","rush_tds",
                "season","week","home","away"]]


def _samples_long(gkey: GameKey, sim_boxes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Convert a list of two‑row team boxes (one per sim) into long samples:
    columns: [season, week, home, away, team, metric, value, sim_id]
    """
    rows = []
    for sid, bx in enumerate(sim_boxes):
        for r in bx.itertuples(index=False):
            for m in CALIB_METRICS:
                rows.append({
                    "season": gkey.season,
                    "week": gkey.week,
                    "home": gkey.home,
                    "away": gkey.away,
                    "team": getattr(r, "team"),
                    "metric": m,
                    "value": float(getattr(r, m)),
                    "sim_id": int(sid),
                })
    return pd.DataFrame(rows)


def simulate_games_for_schedule(
    schedule: pd.DataFrame,
    sims_per_game: int = 200,
    seed: int = 7,
    engine: str = "stateful",
    penalties: bool = False,
    anchor: bool = False,
) -> pd.DataFrame:
    """
    Simulate every game in `schedule` and return a long 'samples' DataFrame.

    Expected schedule columns (uppercase team abbrs recommended):
      - season (int)
      - week   (int)
      - home   (str)
      - away   (str)

    Returns a DataFrame with columns:
      [season, week, home, away, team, metric, value, sim_id]
    where 'metric' ∈ CALIB_METRICS and 'value' is the simulated quantity.
    """
    # Validate schedule shape
    need = {"season", "week", "home", "away"}
    missing = [c for c in need if c not in schedule.columns]
    if missing:
        raise ValueError(f"schedule is missing columns: {missing}")

    # Normalize team codes just in case
    sch = schedule.copy()
    sch["home"] = sch["home"].astype(str).str.upper()
    sch["away"] = sch["away"].astype(str).str.upper()

    # Penalty/anchor knobs to pass through
    penalty_rate = 0.10 if penalties else 0.0
    anchor_weight = 0.25 if anchor else 0.0

    samples_all: List[pd.DataFrame] = []

    for i, row in enumerate(sch.itertuples(index=False), start=1):
        season = int(getattr(row, "season"))
        week   = int(getattr(row, "week"))
        home   = str(getattr(row, "home"))
        away   = str(getattr(row, "away"))

        priors = build_game_priors(home, away, season)

        # Jitter the game seed so different games don't share streams
        game_seed = int(seed + i * 9973)

        if engine == "stateful":
            # The stateful engine runs one sim per call; loop to build a list
            sims: List[pd.DataFrame] = []
            for k in range(int(sims_per_game)):
                df = simulate_stateful(
                    home, away, season,
                    rng=np.random.RandomState(game_seed + 37 * k),
                    priors=priors,
                    penalties_enabled=bool(penalties),
                    penalty_rate=penalty_rate,
                    apply_anchor=bool(anchor),
                    anchor_weight=anchor_weight,
                )
                sims.append(df)
        else:
            # The simple engine returns a list of DataFrames in one call
            sims = simulate_simple(
                home, away, season,
                n_sims=int(sims_per_game),
                seed=game_seed,
                priors=priors,
                penalties_enabled=bool(penalties),
                penalty_rate=penalty_rate,
                anchor_weight=anchor_weight,
            )

        # Convert sims → team game boxes → long samples for calibration
        sim_boxes = [team_box_from_plays(df) for df in sims]
        gkey = GameKey(season=season, week=week, home=home, away=away)
        samples_all.append(_samples_long(gkey, sim_boxes))

    if not samples_all:
        return pd.DataFrame(columns=["season","week","home","away","team","metric","value","sim_id"])

    return pd.concat(samples_all, ignore_index=True)


def summarize_quantiles(
    samples: pd.DataFrame,
    qs: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """
    samples: long table from simulate_games_for_schedule.
    Returns wide 'pred' table keyed by (season, week, home, away, team, metric)
    with columns like q10, q50, q90.
    """
    if samples.empty:
        return pd.DataFrame()

    def _q(x: pd.Series, q: float) -> float:
        return float(np.quantile(x.to_numpy(dtype=float), q))

    grp = samples.groupby(["season","week","home","away","team","metric"])["value"]
    cols = {}
    for q in qs:
        name = f"q{int(round(100*q))}"
        cols[name] = grp.apply(lambda s, q=q: _q(s, q)).reset_index(name=name)

    out = None
    for _, df in cols.items():
        out = df if out is None else out.merge(df, on=["season","week","home","away","team","metric"], how="outer")
    return out.sort_values(["season","week","home","away","team","metric"]).reset_index(drop=True)


def load_truth_csv(path: str) -> pd.DataFrame:
    """
    Expected columns:
      season, week, home, away, team, <metrics...>  (metrics subset may differ; we outer-join)
    You control how to build this truth file (nflverse team box, etc.).
    """
    df = pd.read_csv(path)
    df["home"] = df["home"].str.upper()
    df["away"] = df["away"].str.upper()
    df["team"] = df["team"].str.upper()
    return df


def truth_long(df_truth: pd.DataFrame) -> pd.DataFrame:
    keep = {"season","week","home","away","team"}
    metric_cols = [c for c in df_truth.columns if c not in keep]
    rows = []
    for r in df_truth.itertuples(index=False):
        for m in metric_cols:
            v = getattr(r, m)
            if pd.isna(v):
                continue
            rows.append({
                "season": int(getattr(r, "season")),
                "week": int(getattr(r, "week")),
                "home": str(getattr(r, "home")).upper(),
                "away": str(getattr(r, "away")).upper(),
                "team": str(getattr(r, "team")).upper(),
                "metric": m,
                "truth": float(v),
            })
    return pd.DataFrame(rows)


def reliability_report(
    pred_q: pd.DataFrame,
    truth_long_df: pd.DataFrame,
    qs: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """
    For each metric and quantile q, compute coverage:
      coverage(q) = mean [ truth <= q_pred ]
    """
    if pred_q.empty or truth_long_df.empty:
        return pd.DataFrame(columns=["metric","q","coverage","n"])

    df = pred_q.merge(
        truth_long_df,
        on=["season","week","home","away","team","metric"],
        how="inner"
    )
    rows = []
    for q in qs:
        qcol = f"q{int(round(100*q))}"
        if qcol not in df.columns:
            continue
        sub = df[["metric", qcol, "truth"]].dropna()
        if sub.empty:
            continue
        cov = sub.assign(hit=(sub["truth"] <= sub[qcol]).astype(float)).groupby("metric")["hit"].agg(["mean","count"]).reset_index()
        for r in cov.itertuples(index=False):
            rows.append({"metric": r.metric, "q": float(q), "coverage": float(r.mean), "n": int(r.count)})
    out = pd.DataFrame(rows).sort_values(["metric","q"]).reset_index(drop=True)
    return out


def pit_report(
    samples: pd.DataFrame,
    truth_long_df: pd.DataFrame,
    bins: int = 20,
) -> pd.DataFrame:
    """
    PIT per metric: u = ECDF_sim(truth). For each (game,team,metric), u is in [0,1].
    Returns histogram counts (density) per metric over 'bins'.
    """
    if samples.empty or truth_long_df.empty:
        return pd.DataFrame(columns=["metric","bin_left","bin_right","density","n"])

    # build ECDF per (game, team, metric)
    merged = samples.merge(
        truth_long_df,
        on=["season","week","home","away","team","metric"],
        how="inner",
        suffixes=("", "_truth")
    )
    if merged.empty:
        return pd.DataFrame(columns=["metric","bin_left","bin_right","density","n"])

    # Compute PIT: for each group, share of sim samples <= truth
    # (We avoid O(N^2) by rank trick per group)
    def _pit_group(g: pd.DataFrame) -> float:
        vals = g["value"].to_numpy(dtype=float)
        t = float(g["truth"].iloc[0])
        return float((vals <= t).mean())

    pits = merged.groupby(["season","week","home","away","team","metric"]).apply(_pit_group).reset_index(name="pit")
    pits = pits.dropna(subset=["pit"])

    # Histogram per metric
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for metric, g in pits.groupby("metric"):
        hist, _ = np.histogram(g["pit"].to_numpy(dtype=float), bins=edges, density=True)
        for i in range(bins):
            rows.append({
                "metric": metric,
                "bin_left": float(edges[i]),
                "bin_right": float(edges[i+1]),
                "density": float(hist[i]),
                "n": int(len(g)),
            })
    return pd.DataFrame(rows)


def run_calibration(
    schedule: pd.DataFrame,
    sims_per_game: int,
    seed: int,
    engine: str,
    penalties: bool,
    anchor: bool,
    out_dir: Optional[str] = None,
    truth_csv: Optional[str] = None,
    qs: Sequence[float] = DEFAULT_QUANTILES,
) -> Dict[str, pd.DataFrame]:
    """
    One‑stop driver. Returns dict of DataFrames and optionally writes artifacts.
    """
    samples = simulate_games_for_schedule(
        schedule, sims_per_game=sims_per_game, seed=seed,
        engine=engine, penalties=penalties, anchor=anchor
    )
    pred_q = summarize_quantiles(samples, qs=qs)

    truth_long_df = pd.DataFrame()
    if truth_csv:
        truth = load_truth_csv(truth_csv)
        truth_long_df = truth_long(truth)

    rel = reliability_report(pred_q, truth_long_df, qs=qs) if not truth_long_df.empty else pd.DataFrame()
    pit = pit_report(samples, truth_long_df) if not truth_long_df.empty else pd.DataFrame()

    if out_dir:
        import os
        os.makedirs(out_dir, exist_ok=True)
        samples.to_csv(f"{out_dir}/samples_long.csv.gz", index=False, compression="gzip")
        pred_q.to_csv(f"{out_dir}/pred_quantiles.csv", index=False)
        if not truth_long_df.empty:
            truth_long_df.to_csv(f"{out_dir}/truth_long.csv", index=False)
            rel.to_csv(f"{out_dir}/reliability.csv", index=False)
            pit.to_csv(f"{out_dir}/pit_hist.csv", index=False)

    return {"samples": samples, "pred_quantiles": pred_q, "truth_long": truth_long_df, "reliability": rel, "pit": pit}