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

def load_offense_redzone_profile(year: int, lookback_seasons: int, shrink_k: float = 150.0) -> pd.DataFrame:
    """
    Returns per-offense red-zone TD rate multipliers (phi) vs league:
      - phi_pass_td_rz
      - phi_rush_td_rz
    Using prior `lookback_seasons` regular seasons, with shrinkage to league average.
    """
    import nfl_data_py as nfl
    start = max(2011, year - lookback_seasons)
    years = list(range(start, year))
    if not years:
        years = [year - 1]

    pbp = nfl.import_pbp_data(years)
    df = pbp.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    df["pass"] = df.get("pass", 0).fillna(0).astype(int)
    df["rush"] = df.get("rush", 0).fillna(0).astype(int)
    df["pass_touchdown"] = df.get("pass_touchdown", 0).fillna(0).astype(int)
    df["rush_touchdown"] = df.get("rush_touchdown", 0).fillna(0).astype(int)

    if "yardline_100" not in df.columns:
        raise RuntimeError("yardline_100 not found in pbp; update nfl_data_py.")

    rz = df[df["yardline_100"] <= 20].copy()

    pass_rz = rz[rz["pass"] == 1]
    rush_rz = rz[rz["rush"] == 1]

    p_agg = pass_rz.groupby("posteam").agg(
        pass_att_rz=("pass", "sum"),
        pass_td_rz=("pass_touchdown", "sum"),
    )
    r_agg = rush_rz.groupby("posteam").agg(
        rush_att_rz=("rush", "sum"),
        rush_td_rz=("rush_touchdown", "sum"),
    )

    out = p_agg.join(r_agg, how="outer").fillna(0.0)
    league_pass_att = float(out["pass_att_rz"].sum())
    league_pass_td = float(out["pass_td_rz"].sum())
    league_rush_att = float(out["rush_att_rz"].sum())
    league_rush_td = float(out["rush_td_rz"].sum())

    mu_pass = (league_pass_td / league_pass_att) if league_pass_att > 0 else 0.0
    mu_rush = (league_rush_td / league_rush_att) if league_rush_att > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        shr_pass = (out["pass_td_rz"] + shrink_k * mu_pass) / (out["pass_att_rz"] + shrink_k)
        shr_rush = (out["rush_td_rz"] + shrink_k * mu_rush) / (out["rush_att_rz"] + shrink_k)

    phi_pass = np.where(mu_pass > 0.0, shr_pass / mu_pass, 1.0)
    phi_rush = np.where(mu_rush > 0.0, shr_rush / mu_rush, 1.0)

    out["phi_pass_td_rz"] = np.clip(phi_pass, 0.01, 100.0)
    out["phi_rush_td_rz"] = np.clip(phi_rush, 0.01, 100.0)

    return out[["phi_pass_td_rz", "phi_rush_td_rz"]]


def load_team_scoring_sigma(
    year: int,
    lookback_seasons: int,
    shrink_games: float = 8.0,
    floor: float = 0.80,
    ceil: float = 1.20,
) -> pd.DataFrame:
    """
    Returns per-team sigma multiplier for offensive TD uncertainty (weekly SD),
    shrunk to league average and clipped to [floor, ceil].
    """
    import nfl_data_py as nfl
    start = max(2011, year - lookback_seasons)
    years = list(range(start, year))
    if not years:
        years = [year - 1]

    pbp = nfl.import_pbp_data(years)
    df = pbp.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    df["pass_touchdown"] = df.get("pass_touchdown", 0).fillna(0).astype(int)
    df["rush_touchdown"] = df.get("rush_touchdown", 0).fillna(0).astype(int)

    grp = df.groupby(["season", "game_id", "posteam"]).agg(
        off_tds=("pass_touchdown", "sum")
    )
    grp["off_tds"] += df.groupby(["season", "game_id", "posteam"]).agg(
        add=("rush_touchdown", "sum")
    )["add"]

    # Per-team weekly SD
    team_stats = grp.groupby("posteam")["off_tds"].agg(["std", "count"]).rename(columns={"std": "sd", "count": "n"})
    league_sd = float(grp["off_tds"].std(ddof=1)) if grp["off_tds"].size > 1 else 1.0
    if not np.isfinite(league_sd) or league_sd <= 0:
        league_sd = 1.0

    # Shrink and scale
    shr_sd = (team_stats["sd"] * team_stats["n"] + league_sd * shrink_games) / (team_stats["n"] + shrink_games)
    sigma = shr_sd / league_sd
    sigma = sigma.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    sigma = np.clip(sigma, floor, ceil)

    return sigma.to_frame(name="sigma_td")