from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from .config import SEED, START_SEASON, ROLLING_GAMES, SHARE_ROLLING_GAMES, TEAM_TARGETS, RECEIVING_POS, RUSHING_POS, ARTIFACTS_DIR
from .data import load_pbp, load_weekly_data, load_weekly_rosters, load_injuries, load_schedules, team_schedule_for_year, active_roster_for_week, import_team_list
from .features import build_team_game_stats_from_pbp, build_team_feature_matrix, build_player_share_dataset, compute_player_efficiency
from .models_team import TeamProjectionModel
from .models_shares import ShareProjectionModel
from .utils import ensure_dir, normalize_by_group_sum, allocate_total_by_weights, stable_sort_values
from .data import active_roster_for_week, load_depth_charts
from .config import SHARE_ROLLING_GAMES, RECEIVING_POS, RUSHING_POS, OFFENSE_POS, DEPTH_CHART_LIMITS

def _years_for_training(projection_year: int) -> List[int]:
    return list(range(START_SEASON, projection_year))

def _prep_team_training(projection_year: int) -> pd.DataFrame:
    years = _years_for_training(projection_year)
    pbp = load_pbp(years)
    team_games = build_team_game_stats_from_pbp(pbp)
    feats = build_team_feature_matrix(team_games, window=ROLLING_GAMES)
    return feats

def _prep_share_training(projection_year: int, team_games: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    years = _years_for_training(projection_year)
    weekly = load_weekly_data(years)
    rosters = load_weekly_rosters(years)
    shares = build_player_share_dataset(weekly, team_games, rosters, share_window=SHARE_ROLLING_GAMES)
    eff_player, eff_pos = compute_player_efficiency(weekly)
    return shares, eff_player.merge(eff_pos, on="position", how="left", suffixes=("_player","_pos"))

def _preseason_row_for_team(feats: pd.DataFrame, team: str) -> pd.DataFrame:
    sub = feats[feats["team"] == team].copy()
    if sub.empty:
        # fall back to league average row if team is unseen
        row = feats.tail(1).copy()
        row["team"] = team
        return row.iloc[0:1]
    # last available row (end of last season) already contains rolling means from prior games
    return sub.tail(1)

def _preseason_row_for_opponent(feats: pd.DataFrame, opponent: str) -> pd.DataFrame:
    # we joined opponent rolling defense into the offensive row during feature build; nothing special needed here
    # the preseason row for team already carries joined opponent columns during actual game rows,
    # for preseason we will inject opponent stats by merging later in _assemble_feature_row_for_match.
    return feats[feats["team"] == opponent].tail(1)

def _assemble_feature_row_for_match(team_feats: pd.DataFrame, opp_feats: pd.DataFrame, team: str, opponent: str, is_home: int, projection_year: int) -> pd.DataFrame:
    tr = team_feats[team_feats["team"] == team].copy()
    if tr.empty:
        tr = team_feats.tail(1).copy()
        tr["team"] = team
    tr = tr.tail(1).copy()
    tr["is_home"] = int(is_home)
    tr["season"] = projection_year - 1
    tr["week"] = 0

    opp_rows = team_feats[team_feats["opponent"] == opponent]
    for c in [f"r{ROLLING_GAMES}_opp_allowed_epa_per_play_mean", f"r{ROLLING_GAMES}_opp_allowed_success_rate_mean"]:
        if c in team_feats.columns and not opp_rows.empty:
            tr[c] = opp_rows.iloc[-1][c]

    cols = [col for col in tr.columns if col.startswith(f"r{ROLLING_GAMES}_") or col in ["is_home", "season", "week"]]
    return tr[cols].iloc[0:1]

def train_models(projection_year: int) -> Tuple[TeamProjectionModel, ShareProjectionModel, pd.DataFrame, pd.DataFrame]:
    team_feats = _prep_team_training(projection_year)
    team_model = TeamProjectionModel(projection_year=projection_year, seed=SEED)
    team_model.fit(team_feats)
    share_df, eff = _prep_share_training(projection_year, team_feats)
    share_model = ShareProjectionModel(projection_year=projection_year, seed=SEED)
    share_model.fit(share_df)
    return team_model, share_model, team_feats, eff

from .data import active_roster_for_week, load_depth_charts
from .config import SHARE_ROLLING_GAMES, RECEIVING_POS, RUSHING_POS, OFFENSE_POS, DEPTH_CHART_LIMITS

from .data import active_roster_for_week, load_depth_charts
from .config import SHARE_ROLLING_GAMES, RECEIVING_POS, RUSHING_POS, OFFENSE_POS, DEPTH_CHART_LIMITS

def _roster_features_for_week(team: str, year: int, week: int, player_shares_hist: pd.DataFrame, is_home: int) -> pd.DataFrame:
    roster = active_roster_for_week(team, year, week)
    dc = load_depth_charts([year])
    dc = dc[(dc["team"] == team)].copy()
    if "dt" in dc.columns:
        dc["dt"] = pd.to_datetime(dc["dt"], errors="coerce")
        latest_dt = dc["dt"].max()
        dc = dc[dc["dt"] == latest_dt].copy()
    dc = dc[dc["pos_abb"].isin(OFFENSE_POS)].copy()
    dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    dc = dc[dc["pos_rank"].notna()].copy()
    dc["pos_rank"] = dc["pos_rank"].astype(int)
    def _take_top(g):
        lim = int(DEPTH_CHART_LIMITS.get(g.name, 0))
        if lim <= 0:
            return g.iloc[0:0]
        return g.nsmallest(lim, "pos_rank")
    dc_top = dc.groupby("pos_abb", group_keys=False).apply(_take_top)
    dc_top = dc_top.sort_values(["pos_abb","pos_rank"]).copy()
    if "gsis_id" in dc_top.columns:
        dc_top["player_id"] = dc_top["gsis_id"].where(dc_top["gsis_id"].notna(), dc_top["player_name"])
    else:
        dc_top["player_id"] = dc_top["player_name"]
    dc_top = dc_top.drop_duplicates("player_id")
    dc_top = dc_top.rename(columns={"pos_abb":"position"})[["player_id","player_name","position","pos_rank"]]
    if roster.empty:
        base = dc_top.copy()
        base["season"] = year
        base["week"] = week
        base["team"] = team
        base["age"] = 0.0
        base["years_exp"] = 0.0
        base = base.rename(columns={"pos_rank":"position_rank"})
        roster = base[["season","week","team","player_id","player_name","position","position_rank","age","years_exp"]]
    else:
        roster["position"] = roster.get("position", "").astype(str).str.upper()
        roster = roster[roster["position"].isin(OFFENSE_POS)]
        if "gsis_id" in roster.columns:
            roster["player_id"] = roster["gsis_id"].where(roster["gsis_id"].notna(), roster.get("player_id", roster["player_name"]))
        else:
            roster["player_id"] = roster.get("player_id", roster["player_name"])
        roster = roster.merge(dc_top, on="player_id", how="inner")
        roster = roster.rename(columns={"pos_rank":"position_rank"})
        roster = roster[["season","week","team","player_id","player_name","position","position_rank","age","years_exp"]]
    roster = roster.drop_duplicates("player_id").reset_index(drop=True)
    roster["num_active_same_pos"] = roster.groupby("position")["player_id"].transform("count").astype(float)
    roster["is_home"] = int(is_home)
    hist = player_shares_hist.groupby("player_id").tail(1)[["player_id", f"prev_target_share_{SHARE_ROLLING_GAMES}", f"prev_rush_attempt_share_{SHARE_ROLLING_GAMES}"]]
    roster = roster.merge(hist, on="player_id", how="left")
    roster[f"prev_target_share_{SHARE_ROLLING_GAMES}"] = roster[f"prev_target_share_{SHARE_ROLLING_GAMES}"].fillna(0.0)
    roster[f"prev_rush_attempt_share_{SHARE_ROLLING_GAMES}"] = roster[f"prev_rush_attempt_share_{SHARE_ROLLING_GAMES}"].fillna(0.0)
    return roster

def _apply_efficiency_and_allocate(team_pred: Dict[str,float], shares_df: pd.DataFrame, eff_table: pd.DataFrame) -> pd.DataFrame:
    df = shares_df.copy()
    eff = eff_table.copy().rename(columns={"player_id":"eff_player_id"})
    df = df.merge(eff, left_on="player_id", right_on="eff_player_id", how="left", suffixes=("", "_eff"))
    allowed_ids = set(shares_df["player_id"])
    df = df[df["player_id"].isin(allowed_ids)].copy()
    if "position" not in df.columns:
        if "position_x" in df.columns:
            df["position"] = df["position_x"]
        elif "position_y" in df.columns:
            df["position"] = df["position_y"]
    for metric in ["catch_rate","ypt","rec_td_per_target","yprush","rush_td_per_att"]:
        p = f"{metric}_player"
        q = f"{metric}_pos"
        if p not in df.columns:
            df[p] = np.nan
        if q not in df.columns:
            df[q] = 0.0
        df[p] = df[p].fillna(df[q]).fillna(0.0)
    rx = df[df["position"].isin(RECEIVING_POS)].copy()
    rx["pred_target_share_raw"] = rx["pred_target_share_raw"].fillna(0.0)
    rx_sum = float(rx["pred_target_share_raw"].sum())
    rx["target_share"] = np.where(rx_sum > 0.0, rx["pred_target_share_raw"] / rx_sum, 1.0 / max(1, len(rx)))
    ru = df[df["position"].isin(RUSHING_POS)].copy()
    ru["pred_rush_attempt_share_raw"] = ru["pred_rush_attempt_share_raw"].fillna(0.0)
    ru_sum = float(ru["pred_rush_attempt_share_raw"].sum())
    ru["rush_attempt_share"] = np.where(ru_sum > 0.0, ru["pred_rush_attempt_share_raw"] / ru_sum, 1.0 / max(1, len(ru)))
    out = df[["player_id","player_name","position"] + ([ "position_rank"] if "position_rank" in df.columns else [])].copy()
    out = out.merge(rx[["player_id","target_share"]], on="player_id", how="left")
    out = out.merge(ru[["player_id","rush_attempt_share"]], on="player_id", how="left")
    out["target_share"] = out["target_share"].fillna(0.0)
    out["rush_attempt_share"] = out["rush_attempt_share"].fillna(0.0)
    team_pass_att = float(team_pred.get("pass_attempts", 0.0))
    team_rush_att = float(team_pred.get("rush_attempts", 0.0))
    team_pass_tds = float(team_pred.get("pass_tds", 0.0))
    team_rush_tds = float(team_pred.get("rush_tds", 0.0))
    team_pass_yds = float(team_pred.get("pass_yards", 0.0))
    out = out.merge(
        df[["player_id","catch_rate_player","ypt_player","rec_td_per_target_player","yprush_player","rush_td_per_att_player"]],
        on="player_id",
        how="left"
    ).fillna(0.0)
    out["proj_targets"] = team_pass_att * out["target_share"]
    out["proj_rush_att"] = team_rush_att * out["rush_attempt_share"]
    out["proj_receptions"] = out["proj_targets"] * out["catch_rate_player"]
    out["proj_rec_yards"] = out["proj_targets"] * out["ypt_player"]
    out["proj_rush_yards"] = out["proj_rush_att"] * out["yprush_player"]
    rx_mask = out["position"].isin(RECEIVING_POS)
    weights_rec = (out.loc[rx_mask, "proj_targets"] * out.loc[rx_mask, "rec_td_per_target_player"]).values
    alloc_rec = allocate_total_by_weights(team_pass_tds, weights_rec)
    out.loc[rx_mask, "proj_rec_tds"] = alloc_rec
    ru_mask = out["position"].isin(RUSHING_POS)
    weights_rush = (out.loc[ru_mask, "proj_rush_att"] * out.loc[ru_mask, "rush_td_per_att_player"]).values
    alloc_rush = allocate_total_by_weights(team_rush_tds, weights_rush)
    out.loc[ru_mask, "proj_rush_tds"] = alloc_rush
    out["proj_rec_tds"] = out["proj_rec_tds"].fillna(0.0)
    out["proj_rush_tds"] = out["proj_rush_tds"].fillna(0.0)
    qb_idx = out.index[out["position"] == "QB"]
    out["proj_pass_yards"] = 0.0
    out["proj_pass_tds"] = 0.0
    if len(qb_idx) > 0:
        qb = out.loc[qb_idx].copy()
        if "position_rank" in qb.columns and qb["position_rank"].notna().any():
            min_rank = qb["position_rank"].min()
            weights = (qb["position_rank"] == min_rank).astype(float).values
        else:
            rs = qb["rush_attempt_share"].fillna(0.0).values
            if rs.sum() > 0:
                top = np.zeros_like(rs); top[int(rs.argmax())] = 1.0; weights = top
            else:
                top = np.zeros_like(rs); top[0] = 1.0; weights = top
        alloc_py = allocate_total_by_weights(team_pass_yds, weights)
        alloc_pt = allocate_total_by_weights(team_pass_tds, weights)
        out.loc[qb_idx, "proj_pass_yards"] = alloc_py
        out.loc[qb_idx, "proj_pass_tds"] = alloc_pt
    for c in ["proj_receptions","proj_rec_yards","proj_rush_yards","proj_pass_yards"]:
        out[c] = out[c].clip(lower=0.0)
    keep = [
        "player_id","player_name","position",
        "target_share","rush_attempt_share",
        "proj_targets","proj_receptions","proj_rec_yards","proj_rec_tds",
        "proj_rush_att","proj_rush_yards","proj_rush_tds",
        "proj_pass_yards","proj_pass_tds"
    ]
    return out[keep].sort_values(["position","player_name"]).reset_index(drop=True)

def project_team_week(projection_year: int, team: str, week: int) -> Tuple[pd.DataFrame, Dict[str,float]]:
    team_feats = _prep_team_training(projection_year)
    team_model = TeamProjectionModel(projection_year=projection_year, seed=SEED)
    team_model.fit(team_feats)

    share_df, eff = _prep_share_training(projection_year, team_feats)
    share_model = ShareProjectionModel(projection_year=projection_year, seed=SEED)
    share_model.fit(share_df)

    sched = team_schedule_for_year(team, projection_year)
    wk = sched[sched["week"] == week]
    if wk.empty:
        raise ValueError(f"No regular-season game found for {team} in week {week} of {projection_year}.")
    opp = wk["opponent"].values[0]
    is_home = int(wk["is_home"].values[0])

    team_row = _assemble_feature_row_for_match(team_feats, team_feats, team, opp, is_home, projection_year)
    team_pred = team_model.predict(team_row)

    roster_hist = share_df  # use training constructed history
    roster = _roster_features_for_week(team, projection_year, week, roster_hist, is_home)
    roster["team"] = team
    roster["is_home"] = is_home

    shares_pred = share_model.predict_shares(roster)
    final = _apply_efficiency_and_allocate(team_pred, shares_pred, eff)

    final = final.sort_values(["position","player_name"]).reset_index(drop=True)
    return final, team_pred

def project_team_season(projection_year: int, team: str) -> pd.DataFrame:
    sched = team_schedule_for_year(team, projection_year)
    results = []
    for _, row in sched.iterrows():
        wk = int(row["week"])
        df, team_pred = project_team_week(projection_year, team, wk)
        df["season"] = projection_year
        df["team"] = team
        df["week"] = wk
        for k, v in team_pred.items():
            df[f"team_{k}"] = v
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    return out
