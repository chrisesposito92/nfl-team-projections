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
from .adjust import adjust_team_totals, apply_redzone_td_phi
from .config import RESCALE_EPS
from .data import team_schedule_for_year  # if not already imported

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

    # NEW: compositional share model no longer takes projection_year/seed
    try:
        share_model = ShareProjectionModel.load_or_init()
    except Exception:
        share_model = ShareProjectionModel()
    # keep no-op fit for API compatibility
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

    # NEW: ensure compositional share model can see depth ranks
    if "pos_rank" not in roster.columns and "position_rank" in roster.columns:
        roster["pos_rank"] = roster["position_rank"]
    roster["num_active_same_pos"] = roster.groupby("position")["player_id"].transform("count").astype(float)
    roster["is_home"] = int(is_home)
    hist = player_shares_hist.groupby("player_id").tail(1)[["player_id", f"prev_target_share_{SHARE_ROLLING_GAMES}", f"prev_rush_attempt_share_{SHARE_ROLLING_GAMES}"]]
    roster = roster.merge(hist, on="player_id", how="left")
    roster[f"prev_target_share_{SHARE_ROLLING_GAMES}"] = roster[f"prev_target_share_{SHARE_ROLLING_GAMES}"].fillna(0.0)
    roster[f"prev_rush_attempt_share_{SHARE_ROLLING_GAMES}"] = roster[f"prev_rush_attempt_share_{SHARE_ROLLING_GAMES}"].fillna(0.0)
    return roster

def _apply_efficiency_and_allocate(
    team_pred: Dict[str, float],
    shares_df: pd.DataFrame,
    eff_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine team totals + player shares + efficiency into player-level projections.

    Priority for shares:
      1) 'target_share' / 'rush_attempt_share' if present (already compositional)
      2) 'target_share_pred' / 'rush_attempt_share_pred' (scores -> normalize)
      3) 'pred_target_share_raw' / 'pred_rush_attempt_share_raw' (scores -> normalize)
      4) uniform within eligibility group as last resort
    """
    df = shares_df.copy()

    # --- merge efficiency (player, fallback to position) ---
    eff = eff_table.copy().rename(columns={"player_id": "eff_player_id"})
    df = df.merge(
        eff,
        left_on="player_id",
        right_on="eff_player_id",
        how="left",
        suffixes=("", "_eff"),
    )

    # Keep only requested players (paranoia if joins created extras)
    allowed_ids = set(shares_df["player_id"])
    df = df[df["player_id"].isin(allowed_ids)].copy()

    # Ensure we have a 'position' column (some merges create position_x/y)
    if "position" not in df.columns:
        if "position_x" in df.columns:
            df["position"] = df["position_x"]
        elif "position_y" in df.columns:
            df["position"] = df["position_y"]

    # Fill efficiency using player first, then position aggregates, then 0.0
    for metric in ["catch_rate", "ypt", "rec_td_per_target", "yprush", "rush_td_per_att"]:
        p = f"{metric}_player"
        q = f"{metric}_pos"
        if p not in df.columns:
            df[p] = np.nan
        if q not in df.columns:
            df[q] = 0.0
        df[p] = df[p].fillna(df[q]).fillna(0.0)

    # --- build shares with fallbacks ---

    # Receiving eligibility: RB/WR/TE (from config.RECEIVING_POS)
    rx = df[df["position"].isin(RECEIVING_POS)].copy()
    if len(rx) > 0:
        if "target_share" in rx.columns and rx["target_share"].notna().any():
            s = rx["target_share"].fillna(0.0).to_numpy(float)
        elif "target_share_pred" in rx.columns:
            raw = rx["target_share_pred"].fillna(0.0).to_numpy(float)
            z = raw.sum()
            s = raw / z if z > 0 else np.full(len(rx), 1.0 / len(rx))
        elif "pred_target_share_raw" in rx.columns:
            raw = rx["pred_target_share_raw"].fillna(0.0).to_numpy(float)
            z = raw.sum()
            s = raw / z if z > 0 else np.full(len(rx), 1.0 / len(rx))
        else:
            s = np.full(len(rx), 1.0 / len(rx))
        # exact renorm/clip for safety
        s = np.clip(s, 0.0, 1.0)
        z = s.sum()
        s = s / z if z > 0 else np.full(len(rx), 1.0 / len(rx))
        rx["target_share"] = s

    # Rushing eligibility: RB/QB (from config.RUSHING_POS)
    ru = df[df["position"].isin(RUSHING_POS)].copy()
    if len(ru) > 0:
        if "rush_attempt_share" in ru.columns and ru["rush_attempt_share"].notna().any():
            s = ru["rush_attempt_share"].fillna(0.0).to_numpy(float)
        elif "rush_attempt_share_pred" in ru.columns:
            raw = ru["rush_attempt_share_pred"].fillna(0.0).to_numpy(float)
            z = raw.sum()
            s = raw / z if z > 0 else np.full(len(ru), 1.0 / len(ru))
        elif "pred_rush_attempt_share_raw" in ru.columns:
            raw = ru["pred_rush_attempt_share_raw"].fillna(0.0).to_numpy(float)
            z = raw.sum()
            s = raw / z if z > 0 else np.full(len(ru), 1.0 / len(ru))
        else:
            s = np.full(len(ru), 1.0 / len(ru))
        # exact renorm/clip for safety
        s = np.clip(s, 0.0, 1.0)
        z = s.sum()
        s = s / z if z > 0 else np.full(len(ru), 1.0 / len(ru))
        ru["rush_attempt_share"] = s

    # Build output scaffold
    out_cols = ["player_id", "player_name", "position"]
    if "position_rank" in df.columns:
        out_cols.append("position_rank")
    out = df[out_cols].copy()

    # Attach shares (fill missing with 0.0)
    if len(rx) > 0:
        out = out.merge(rx[["player_id", "target_share"]], on="player_id", how="left")
    else:
        out["target_share"] = 0.0

    if len(ru) > 0:
        out = out.merge(ru[["player_id", "rush_attempt_share"]], on="player_id", how="left")
    else:
        out["rush_attempt_share"] = 0.0

    out["target_share"] = out["target_share"].fillna(0.0)
    out["rush_attempt_share"] = out["rush_attempt_share"].fillna(0.0)

    # Team totals
    team_pass_att = float(team_pred.get("pass_attempts", 0.0))
    team_rush_att = float(team_pred.get("rush_attempts", 0.0))
    team_pass_tds = float(team_pred.get("pass_tds", 0.0))
    team_rush_tds = float(team_pred.get("rush_tds", 0.0))
    team_pass_yds = float(team_pred.get("pass_yards", 0.0))

    # Merge efficiencies we prepared above
    out = out.merge(
        df[
            [
                "player_id",
                "catch_rate_player",
                "ypt_player",
                "rec_td_per_target_player",
                "yprush_player",
                "rush_td_per_att_player",
            ]
        ],
        on="player_id",
        how="left",
    ).fillna(0.0)

    # Deterministic projections from shares + efficiency
    out["proj_targets"] = team_pass_att * out["target_share"]
    out["proj_rush_att"] = team_rush_att * out["rush_attempt_share"]

    out["proj_receptions"] = out["proj_targets"] * out["catch_rate_player"]
    out["proj_rec_yards"] = out["proj_targets"] * out["ypt_player"]
    out["proj_rush_yards"] = out["proj_rush_att"] * out["yprush_player"]

    # Allocate receiving TDs across receiving-eligible positions
    rx_mask = out["position"].isin(RECEIVING_POS)
    weights_rec = (
        out.loc[rx_mask, "proj_targets"].to_numpy(float)
        * out.loc[rx_mask, "rec_td_per_target_player"].to_numpy(float)
    )
    alloc_rec = allocate_total_by_weights(team_pass_tds, weights_rec)
    out.loc[rx_mask, "proj_rec_tds"] = alloc_rec

    # Allocate rushing TDs across rushing-eligible positions
    ru_mask = out["position"].isin(RUSHING_POS)
    weights_rush = (
        out.loc[ru_mask, "proj_rush_att"].to_numpy(float)
        * out.loc[ru_mask, "rush_td_per_att_player"].to_numpy(float)
    )
    alloc_rush = allocate_total_by_weights(team_rush_tds, weights_rush)
    out.loc[ru_mask, "proj_rush_tds"] = alloc_rush

    out["proj_rec_tds"] = out["proj_rec_tds"].fillna(0.0)
    out["proj_rush_tds"] = out["proj_rush_tds"].fillna(0.0)

    # Attribute pass yards/TDs to QB(s)
    qb_idx = out.index[out["position"] == "QB"]
    out["proj_pass_yards"] = 0.0
    out["proj_pass_tds"] = 0.0
    if len(qb_idx) > 0:
        qb = out.loc[qb_idx].copy()
        if "position_rank" in qb.columns and qb["position_rank"].notna().any():
            min_rank = qb["position_rank"].min()
            weights = (qb["position_rank"] == min_rank).astype(float).to_numpy()
        else:
            # Fallback: lean toward the QB with more designed+rushing share if available
            rs = qb["rush_attempt_share"].fillna(0.0).to_numpy(float)
            if rs.sum() > 0:
                weights = np.zeros_like(rs)
                weights[int(rs.argmax())] = 1.0
            else:
                weights = np.zeros_like(rs)
                weights[0] = 1.0

        alloc_py = allocate_total_by_weights(team_pass_yds, weights)
        alloc_pt = allocate_total_by_weights(team_pass_tds, weights)
        out.loc[qb_idx, "proj_pass_yards"] = alloc_py
        out.loc[qb_idx, "proj_pass_tds"] = alloc_pt

    # Non-negative guards
    for c in ["proj_receptions", "proj_rec_yards", "proj_rush_yards", "proj_pass_yards"]:
        out[c] = out[c].clip(lower=0.0)

    keep = [
        "player_id",
        "player_name",
        "position",
        "target_share",
        "rush_attempt_share",
        "proj_targets",
        "proj_receptions",
        "proj_rec_yards",
        "proj_rec_tds",
        "proj_rush_att",
        "proj_rush_yards",
        "proj_rush_tds",
        "proj_pass_yards",
        "proj_pass_tds",
    ]
    return out[keep].sort_values(["position", "player_name"]).reset_index(drop=True)

def project_team_week(projection_year: int, team: str, week: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    team_feats = _prep_team_training(projection_year)
    team_model = TeamProjectionModel(projection_year=projection_year, seed=SEED)
    team_model.fit(team_feats)

    share_df, eff = _prep_share_training(projection_year, team_feats)
    # NEW: load/init compositional share model
    try:
        share_model = ShareProjectionModel.load_or_init()
    except Exception:
        share_model = ShareProjectionModel()
    share_model.fit(share_df)

    sched = team_schedule_for_year(team, projection_year)
    wk = sched[sched["week"] == week]
    if wk.empty:
        raise ValueError(f"No regular-season game found for {team} in week {week} of {projection_year}.")
    opp = str(wk["opponent"].iloc[0])
    is_home_int = int(wk["is_home"].iloc[0])
    is_home_bool = bool(is_home_int == 1)

    team_row = _assemble_feature_row_for_match(team_feats, team_feats, team, opp, is_home_int, projection_year)
    team_pred = team_model.predict(team_row)

    team_pred = adjust_team_totals(team_pred, opponent=opp, is_home=is_home_bool, year=projection_year)
    team_pred = apply_redzone_td_phi(team_pred, team=team, year=projection_year)

    roster_hist = share_df
    roster = _roster_features_for_week(team, projection_year, week, roster_hist, is_home_int)
    roster["team"] = team
    roster["is_home"] = is_home_int

    shares_pred = share_model.predict_shares(roster)
    final = _apply_efficiency_and_allocate(team_pred, shares_pred, eff)
    final = _enforce_identity_constraints(final, team_pred)

    final = final.sort_values(["position", "player_name"]).reset_index(drop=True)
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

def _enforce_identity_constraints(df: pd.DataFrame, team_pred: dict) -> pd.DataFrame:
    """
    Rescales player columns so they sum exactly to team totals.
    - Receiving yards/TDs across WR/RB/TE sum to team pass_yards/pass_tds.
    - Rushing yards/TDs across QB/RB/WR/TE sum to team rush_yards/rush_tds.
    """
    def _rescale(mask: pd.Series, col: str, target: float):
        s = float(df.loc[mask, col].sum())
        if s > RESCALE_EPS and target is not None and np.isfinite(target):
            df.loc[mask, col] = df.loc[mask, col] * (float(target) / s)
        else:
            df.loc[mask, col] = 0.0

    rx_mask = df["position"].isin(["WR", "TE", "RB"])
    rush_mask = df["position"].isin(["QB", "RB", "WR", "TE"])

    _rescale(rx_mask, "proj_rec_yards", team_pred.get("pass_yards", 0.0))
    _rescale(rx_mask, "proj_rec_tds",   team_pred.get("pass_tds", 0.0))
    _rescale(rush_mask, "proj_rush_yards", team_pred.get("rush_yards", 0.0))
    _rescale(rush_mask, "proj_rush_tds",   team_pred.get("rush_tds", 0.0))

    return df