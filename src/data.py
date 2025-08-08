from __future__ import annotations
import pandas as pd
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
