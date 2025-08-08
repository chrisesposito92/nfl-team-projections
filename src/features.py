from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple
from .utils import stable_sort_values
from .config import RECEIVING_POS, RUSHING_POS

def build_team_game_stats_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    if "epa" not in df.columns:
        df["epa"] = 0.0
    for c in ["passing_yards", "rushing_yards", "pass_touchdown", "rush_touchdown"]:
        if c not in df.columns:
            df[c] = 0.0
    if "pass_attempt" not in df.columns:
        df["pass_attempt"] = df["pass"].fillna(0).astype(int) if "pass" in df.columns else 0
    if "rush_attempt" not in df.columns:
        df["rush_attempt"] = df["rush"].fillna(0).astype(int) if "rush" in df.columns else 0

    df = df[df["posteam"].notna() & df["defteam"].notna()].copy()
    df["success"] = (df["epa"] > 0).astype(int)

    g = df.groupby(["game_id", "season", "week", "posteam", "defteam"], dropna=False).agg(
        pass_attempts=("pass_attempt", "sum"),
        pass_yards=("passing_yards", "sum"),
        pass_tds=("pass_touchdown", "sum"),
        rush_attempts=("rush_attempt", "sum"),
        rush_yards=("rushing_yards", "sum"),
        rush_tds=("rush_touchdown", "sum"),
        epa_offense=("epa", "mean"),
        success_offense=("success", "mean"),
        plays=("epa", "count"),
    ).reset_index().rename(columns={"posteam": "team", "defteam": "opponent"})

    g["is_home"] = 0
    if "home_team" in df.columns:
        home_map = df[["game_id", "home_team"]].dropna().drop_duplicates("game_id")
        g = g.merge(home_map, on="game_id", how="left")
        g["is_home"] = (g["team"] == g["home_team"]).astype(int)
        g = g.drop(columns=[c for c in ["home_team"] if c in g.columns])

    dg = df.groupby(["game_id", "season", "week", "defteam"], dropna=False).agg(
        allowed_pass_attempts=("pass_attempt", "sum"),
        allowed_pass_yards=("passing_yards", "sum"),
        allowed_pass_tds=("pass_touchdown", "sum"),
        allowed_rush_attempts=("rush_attempt", "sum"),
        allowed_rush_yards=("rushing_yards", "sum"),
        allowed_rush_tds=("rush_touchdown", "sum"),
        allowed_epa_per_play=("epa", "mean"),
        allowed_success_rate=("epa", lambda s: (s > 0).mean()),
        d_plays=("epa", "count"),
    ).reset_index().rename(columns={"defteam": "def_team"})

    out = g.merge(
        dg[["game_id", "season", "week", "def_team", "allowed_epa_per_play", "allowed_success_rate"]],
        left_on=["game_id", "season", "week", "opponent"],
        right_on=["game_id", "season", "week", "def_team"],
        how="left"
    ).drop(columns=["def_team"])

    return stable_sort_values(out, ["season", "week", "team"])

def _rolling_means(df: pd.DataFrame, group_key: str, order_cols: List[str], cols: List[str], window: int) -> pd.DataFrame:
    df = df.copy()
    df = stable_sort_values(df, [group_key] + order_cols)
    for col in cols:
        df[f"r{window}_{col}_mean"] = (
            df.groupby(group_key, group_keys=False)[col]
              .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
    return df

def build_team_feature_matrix(team_games: pd.DataFrame, window: int) -> pd.DataFrame:
    off_cols = ["pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds", "epa_offense", "success_offense"]
    tmp = _rolling_means(team_games, "team", ["season", "week"], off_cols, window)

    def_cols = ["allowed_epa_per_play", "allowed_success_rate"]
    dview = team_games[["season", "week", "opponent"] + def_cols].copy()
    dview = _rolling_means(dview, "opponent", ["season", "week"], def_cols, window)
    dview = dview.rename(columns={
        f"r{window}_{c}_mean": f"r{window}_opp_{c}_mean" for c in def_cols
    })

    feat = tmp.merge(
        dview[["season", "week", "opponent", f"r{window}_opp_allowed_epa_per_play_mean", f"r{window}_opp_allowed_success_rate_mean"]],
        on=["season", "week", "opponent"],
        how="left"
    )
    return stable_sort_values(feat, ["season", "week", "team"])

def build_player_share_dataset(weekly: pd.DataFrame, team_games: pd.DataFrame, rosters: pd.DataFrame, share_window: int) -> pd.DataFrame:
    w = weekly.copy()
    if "team" not in w.columns and "recent_team" in w.columns:
        w["team"] = w["recent_team"]

    def coalesce(df: pd.DataFrame, name: str, candidates: List[str], fill=0.0):
        if name in df.columns:
            return
        for c in candidates:
            if c in df.columns:
                df[name] = df[c]
                return
        df[name] = fill

    coalesce(w, "targets", ["targets"])
    coalesce(w, "receptions", ["receptions"])
    coalesce(w, "receiving_yards", ["receiving_yards", "rec_yards"])
    coalesce(w, "receiving_tds", ["receiving_tds", "rec_tds"])
    coalesce(w, "player_rush_attempts", ["rushing_attempts", "rush_attempts", "rushing_att", "carries"])
    coalesce(w, "rushing_yards", ["rushing_yards"])
    coalesce(w, "rushing_tds", ["rushing_tds", "rush_tds"])

    for c in ["season","week","player_id","player_name","team","position"]:
        if c not in w.columns:
            w[c] = "" if c in ["player_id","player_name","team","position"] else 0

    # Keep offense only
    offense = set(RUSHING_POS) | set(RECEIVING_POS)
    w = w[w["position"].isin(offense)].copy()

    den = team_games[["season","week","team","pass_attempts","rush_attempts"]].copy()
    den = den.rename(columns={"pass_attempts":"team_pass_attempts","rush_attempts":"team_rush_attempts"})
    w = w.merge(den, on=["season","week","team"], how="left", validate="many_to_one")

    w["team_pass_attempts"] = w["team_pass_attempts"].fillna(0.0)
    w["team_rush_attempts"] = w["team_rush_attempts"].fillna(0.0)
    w["targets"] = w["targets"].fillna(0.0)
    w["player_rush_attempts"] = w["player_rush_attempts"].fillna(0.0)

    w["target_share"] = np.where(w["team_pass_attempts"] > 0, w["targets"] / w["team_pass_attempts"], 0.0)
    w["rush_attempt_share"] = np.where(w["team_rush_attempts"] > 0, w["player_rush_attempts"] / w["team_rush_attempts"], 0.0)

    w = stable_sort_values(w, ["player_id","season","week"])
    for col in ["target_share","rush_attempt_share"]:
        w[f"prev_{col}_{share_window}"] = (
            w.groupby("player_id", group_keys=False)[col]
             .apply(lambda s: s.shift(1).rolling(share_window, min_periods=1).mean())
             .fillna(0.0)
        )

    r = rosters.copy()
    if "team" not in r.columns and "recent_team" in r.columns:
        r["team"] = r["recent_team"]
    for k in ["season","week","team","player_id","age","years_exp"]:
        if k not in r.columns:
            r[k] = np.nan
    r = r[["season","week","team","player_id","age","years_exp"]].copy()

    w = w.merge(r, on=["season","week","team","player_id"], how="left")
    w["age"] = w["age"].fillna(w.groupby("player_id")["age"].transform("max"))
    w["years_exp"] = w["years_exp"].fillna(w.groupby("player_id")["years_exp"].transform("max"))
    w["age"] = w["age"].fillna(0.0)
    w["years_exp"] = w["years_exp"].fillna(0.0)

    counts = w.groupby(["season","week","team","position"], dropna=False)["player_id"].transform("count")
    w["num_active_same_pos"] = counts.fillna(1.0)

    g = team_games[["season","week","team","is_home"]].drop_duplicates()
    w = w.merge(g, on=["season","week","team"], how="left")
    w["is_home"] = w["is_home"].fillna(0).astype(int)

    use_cols = ["season","week","team","player_id","player_name","position","age","years_exp",
                "num_active_same_pos","is_home","target_share","rush_attempt_share",
                f"prev_target_share_{share_window}", f"prev_rush_attempt_share_{share_window}"]
    return w[use_cols].copy()

def compute_player_efficiency(weekly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w = weekly.copy()
    if "team" not in w.columns and "recent_team" in w.columns:
        w["team"] = w["recent_team"]

    def coalesce(df: pd.DataFrame, name: str, candidates: List[str], fill=0.0):
        if name in df.columns:
            return
        for c in candidates:
            if c in df.columns:
                df[name] = df[c]
                return
        df[name] = fill

    coalesce(w, "targets", ["targets"])
    coalesce(w, "receptions", ["receptions"])
    coalesce(w, "receiving_yards", ["receiving_yards", "rec_yards"])
    coalesce(w, "receiving_tds", ["receiving_tds", "rec_tds"])
    coalesce(w, "rushing_attempts", ["rushing_attempts", "rush_attempts", "rushing_att", "carries"])
    coalesce(w, "rushing_yards", ["rushing_yards"])
    coalesce(w, "rushing_tds", ["rushing_tds", "rush_tds"])
    for c in ["position","player_id"]:
        if c not in w.columns:
            w[c] = ""

    offense = set(RUSHING_POS) | set(RECEIVING_POS)
    w = w[w["position"].isin(offense)].copy()

    grp = w.groupby(["player_id","position"], dropna=False).agg(
        tot_targets=("targets","sum"),
        tot_receptions=("receptions","sum"),
        tot_rec_yards=("receiving_yards","sum"),
        tot_rec_tds=("receiving_tds","sum"),
        tot_rush_att=("rushing_attempts","sum"),
        tot_rush_yards=("rushing_yards","sum"),
        tot_rush_tds=("rushing_tds","sum"),
    ).reset_index()

    grp["catch_rate"] = np.where(grp["tot_targets"]>0, grp["tot_receptions"]/grp["tot_targets"], 0.0)
    grp["ypt"] = np.where(grp["tot_targets"]>0, grp["tot_rec_yards"]/grp["tot_targets"], 0.0)
    grp["rec_td_per_target"] = np.where(grp["tot_targets"]>0, grp["tot_rec_tds"]/grp["tot_targets"], 0.0)
    grp["yprush"] = np.where(grp["tot_rush_att"]>0, grp["tot_rush_yards"]/grp["tot_rush_att"], 0.0)
    grp["rush_td_per_att"] = np.where(grp["tot_rush_att"]>0, grp["tot_rush_tds"]/grp["tot_rush_att"], 0.0)

    pos = grp.groupby("position").agg(
        catch_rate=("catch_rate","mean"),
        ypt=("ypt","mean"),
        rec_td_per_target=("rec_td_per_target","mean"),
        yprush=("yprush","mean"),
        rush_td_per_att=("rush_td_per_att","mean"),
    ).reset_index()

    return grp, pos
