from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from .utils import ensure_dir, stable_sort_values

def _years_list(start: int, end: int) -> List[int]:
    return list(range(start, end + 1))

def import_team_list():
    import nfl_data_py as nfl
    teams = nfl.import_team_desc()
    teams = teams[teams["team_abbr"].notna()].copy()
    teams = teams.sort_values("team_abbr")
    return teams["team_abbr"].unique().tolist()

def load_schedules(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    df = nfl.import_schedules(years)
    df = df[df["game_type"] == "REG"].copy()
    # normalize columns expected later
    keep = ["game_id","season","week","home_team","away_team","gameday","game_type"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA
    df = df[keep].copy()
    df.rename(columns={"gameday": "game_date"}, inplace=True)
    return df

def load_pbp(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    df = nfl.import_pbp_data(years, downcast=False)
    # keep only regular season games
    # join to schedules for date
    sched = load_schedules(years)
    df = df.merge(sched[["game_id","season","week","game_date","home_team","away_team"]].drop_duplicates("game_id"),
                  on=["game_id","season","week"], how="left")
    return df

def load_weekly_data(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    df = nfl.import_weekly_data(years)
    return df

def load_snap_counts(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        df = nfl.import_snap_counts(years)
    except Exception:
        df = pd.DataFrame()
    return df

def load_weekly_rosters(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        df = nfl.import_weekly_rosters(years)
    except Exception:
        cols = [
            "season","week","team","position","player_id","player_name","gsis_id",
            "age","years_exp","status","depth_chart_position","jersey_number","college"
        ]
        df = pd.DataFrame(columns=cols)
    return df

def load_injuries(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        df = nfl.import_injuries(years)
    except Exception:
        cols = ["season","week","game_status","player_id","gsis_id"]
        df = pd.DataFrame(columns=cols)
    return df

def team_schedule_for_year(team: str, year: int) -> pd.DataFrame:
    sched = load_schedules([year])
    mask = (sched["home_team"] == team) | (sched["away_team"] == team)
    team_sched = sched.loc[mask].copy()
    team_sched["opponent"] = team_sched.apply(lambda r: r["away_team"] if r["home_team"] == team else r["home_team"], axis=1)
    team_sched["is_home"] = (team_sched["home_team"] == team).astype(int)
    team_sched = stable_sort_values(team_sched, ["week"])
    return team_sched[["season","week","game_id","home_team","away_team","opponent","is_home","game_date"]].reset_index(drop=True)

def active_roster_for_week(team: str, year: int, week: int) -> pd.DataFrame:
    rosters = load_weekly_rosters([year])
    players = rosters[(rosters["team"] == team) & (rosters["season"] == year) & (rosters["week"] == week)].copy()
    injuries = load_injuries([year])
    if not injuries.empty:
        outs = injuries[(injuries["season"] == year) & (injuries["week"] == week)]
        outs = outs[outs["game_status"].isin(["Out","Injured Reserve"])]
        if "gsis_id" in outs.columns and "gsis_id" in players.columns:
            players = players[~players["gsis_id"].isin(outs["gsis_id"])]
        elif "player_id" in outs.columns and "player_id" in players.columns:
            players = players[~players["player_id"].isin(outs["player_id"])]
    cols = ["season","week","team","position","player_id","player_name","gsis_id","birth_date","height","weight","status","depth_chart_position","jersey_number","college","years_exp","age"]
    keep = [c for c in cols if c in players.columns]
    return players[keep].copy()

def load_depth_charts(years: List[int]) -> pd.DataFrame:
    import nfl_data_py as nfl
    try:
        df = nfl.import_depth_charts(years)
    except Exception:
        cols = ["dt","team","player_name","espn_id","gsis_id","pos_grp_id","pos_grp","pos_id","pos_name","pos_abb","pos_slot","pos_rank"]
        df = pd.DataFrame(columns=cols)
    return df

def load_defense_profile(year: int, lookback_seasons: int) -> pd.DataFrame:
    """
    Build opponent defensive profile from prior seasons' play-by-play:
    - pass_epa_allowed, rush_epa_allowed
    - neutral_pass_rate_allowed (1st/2nd down, Q1-3, score diff within +/-7)
    Returns a DataFrame indexed by defense team abbr with z-scored columns:
      pass_epa_allowed_z, rush_epa_allowed_z, neutral_pass_rate_allowed_z
    """
    import nfl_data_py as nfl

    start = max(2011, year - lookback_seasons)
    years = list(range(start, year))  # prior seasons only
    if not years:
        years = [year - 1]

    pbp = nfl.import_pbp_data(years)
    df = pbp.copy()

    # Ensure boolean flags exist/are numeric
    if "pass" in df.columns:
        df["pass"] = df["pass"].fillna(0).astype(int)
    if "rush" in df.columns:
        df["rush"] = df["rush"].fillna(0).astype(int)

    # Keep only rush/pass plays
    mask = (df.get("pass", 0) == 1) | (df.get("rush", 0) == 1)
    df = df.loc[mask].copy()

    # Pass and rush splits
    pass_df = df[df["pass"] == 1].copy()
    rush_df = df[df["rush"] == 1].copy()

    # Neutral pass rate allowed (defensive view)
    neutral = df[
        df["down"].isin([1, 2])
        & (df["qtr"] <= 3)
        & (df["score_differential"].between(-7, 7))
    ].copy()
    neutral_agg = neutral.groupby("defteam").agg(
        pass_plays=("pass", "sum"),
        rush_plays=("rush", "sum"),
    )
    neutral_agg["neutral_pass_rate_allowed"] = neutral_agg["pass_plays"] / (
        neutral_agg["pass_plays"] + neutral_agg["rush_plays"]
    )

    # Explosive not used in v1 adjustments, but kept if you want to extend
    if "yards_gained" in pass_df.columns:
        pass_df["explosive"] = (pass_df["yards_gained"] >= 20).astype(float)

    out = pd.DataFrame(index=sorted(set(df["defteam"].dropna().unique())))
    out["pass_epa_allowed"] = pass_df.groupby("defteam")["epa"].mean()
    out["rush_epa_allowed"] = rush_df.groupby("defteam")["epa"].mean()
    out = out.join(neutral_agg["neutral_pass_rate_allowed"])

    out = out.replace([np.inf, -np.inf], np.nan)

    # z-scores for adjustments
    for col in ["pass_epa_allowed", "rush_epa_allowed", "neutral_pass_rate_allowed"]:
        mu = out[col].mean()
        sd = out[col].std(ddof=0)
        if sd and sd > 0:
            out[col + "_z"] = (out[col] - mu) / sd
        else:
            out[col + "_z"] = 0.0
        out[col + "_z"] = out[col + "_z"].fillna(0.0)

    return out