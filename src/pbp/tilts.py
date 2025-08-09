from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional
import numpy as np
import pandas as pd

# Only hard-require depth charts; if not available, stub to empty
try:
    from ..data import load_depth_charts
except Exception:
    def load_depth_charts(years):
        return pd.DataFrame()

from ..config import RECEIVING_POS, RUSHING_POS, OFFENSE_POS
from ..models_shares import ShareProjectionModel


def _zscore(a: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    m = a.mean()
    s = a.std(ddof=0)
    if not np.isfinite(s) or s <= 1e-12:
        return pd.Series(np.zeros(len(a)), index=a.index)
    return (a - m) / s


def _latest_depth(team: str, year: int) -> pd.DataFrame:
    dc = load_depth_charts([year])
    dc = dc[(dc.get("team") == team) & (dc.get("pos_abb").isin(OFFENSE_POS))].copy()
    if "dt" in dc.columns:
        dc["dt"] = pd.to_datetime(dc["dt"], errors="coerce")
        dc = dc[dc["dt"] == dc["dt"].max()].copy()
    dc["pos_rank"] = pd.to_numeric(dc.get("pos_rank"), errors="coerce")
    dc = dc.dropna(subset=["pos_rank"])
    dc["pos_rank"] = dc["pos_rank"].astype(int)
    if "gsis_id" in dc.columns:
        dc["player_id"] = dc["gsis_id"].where(dc["gsis_id"].notna(), dc["player_name"])
    else:
        dc["player_id"] = dc["player_name"]
    return (
        dc.rename(columns={"pos_abb": "position"})
          [["player_id", "player_name", "position", "pos_rank"]]
          .drop_duplicates("player_id")
    )


def _compose_base_shares(team: str, year: int) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tuple[str, str]]]:
    """
    Build base compositional shares (targets, rush attempts) from depth chart
    using the same share model you use elsewhere.
    Returns: (target_share_by_id, rush_share_by_id, meta_by_id)
    meta_by_id[player_id] -> (player_name, position)
    """
    dc = _latest_depth(team, year)
    if dc.empty:
        return {}, {}, {}

    roster = dc.copy()
    roster["team"] = team

    sm = ShareProjectionModel.load_or_init()
    shares = sm.predict_shares(roster)

    tgt: Dict[str, float] = {}
    ru: Dict[str, float] = {}
    meta: Dict[str, Tuple[str, str]] = {}
    for r in shares.itertuples(index=False):
        pid = getattr(r, "player_id")
        pos = getattr(r, "position")
        meta[pid] = (getattr(r, "player_name"), pos)
        tgt[pid] = float(getattr(r, "target_share")) if pos in RECEIVING_POS else 0.0
        ru[pid]  = float(getattr(r, "rush_attempt_share")) if pos in RUSHING_POS else 0.0

    def _norm(d: Dict[str, float], elig_positions: Iterable[str]) -> Dict[str, float]:
        ids = [k for k, (_, p) in meta.items() if p in elig_positions]
        s = sum(d.get(i, 0.0) for i in ids)
        if s > 0:
            for i in ids:
                d[i] = d.get(i, 0.0) / s
        return d

    tgt = _norm(tgt, RECEIVING_POS)
    ru = _norm(ru, RUSHING_POS)
    return tgt, ru, meta


def load_ngs_tilts_for_game(home: str, away: str, year: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build per-team, per-player tilts from Next Gen Stats.
    Returns:
      {
        "rec_target": {TEAM: {player_id: tilt}},
        "rec_complete": {TEAM: {player_id: tilt}},
        "rush_carry": {TEAM: {player_id: tilt}},
      }
    Tilts are bounded ~[0.5, 1.5] for stability.
    If NGS loaders aren't available, returns empty dicts (neutral tilts = 1.0 downstream).
    """
    teams = [home, away]

    # Import here so package import doesn't hard-require these loaders
    try:
        from ..data import import_ngs_data  # type: ignore
    except Exception:
        import_ngs_data = None  # type: ignore

    if import_ngs_data is None:
        return {"rec_target": {t: {} for t in teams},
                "rec_complete": {t: {} for t in teams},
                "rush_carry": {t: {} for t in teams}}

    try:
        ngs_rx = import_ngs_data("receiving", [year])
    except Exception:
        ngs_rx = pd.DataFrame()

    try:
        ngs_ru = import_ngs_data("rushing", [year])
    except Exception:
        ngs_ru = pd.DataFrame()

    def _prep_ids(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["player_id"] = df.get("player_gsis_id")
        if "player_id" not in df.columns or df["player_id"].isna().all():
            df["player_id"] = df.get("player_display_name")
        return df

    ngs_rx = _prep_ids(ngs_rx)
    ngs_ru = _prep_ids(ngs_ru)

    rec_target = {t: {} for t in teams}
    rec_complete = {t: {} for t in teams}
    rush_carry = {t: {} for t in teams}

    # Receiving tilts
    if not ngs_rx.empty:
        sub = ngs_rx[(ngs_rx["season"] == year)]
        for t in teams:
            g = sub[sub["team_abbr"] == t].copy()
            if g.empty:
                continue
            agg = g.groupby(["player_id"], as_index=False).agg({
                "avg_separation": "mean",
                "avg_intended_air_yards": "mean",
                "percent_share_of_intended_air_yards": "mean",
                "catch_percentage": "mean",
            })
            z_sep  = _zscore(agg["avg_separation"])
            z_iay  = _zscore(agg["avg_intended_air_yards"])
            z_iay_share = _zscore(agg["percent_share_of_intended_air_yards"])
            z_catch = _zscore(agg["catch_percentage"])

            target_raw = 0.30*z_sep + 0.25*z_iay + 0.25*z_iay_share + 0.10*z_catch
            comp_raw   = 0.50*z_catch + 0.20*z_sep - 0.15*z_iay  # deeper IAY -> lower comp

            for i, r in agg.iterrows():
                pid = r["player_id"]
                rec_target[t][pid] = float(np.clip(np.exp(target_raw.iloc[i]), 0.5, 1.5))
                rec_complete[t][pid] = float(np.clip(np.exp(comp_raw.iloc[i]), 0.5, 1.5))

    # Rushing tilts
    if not ngs_ru.empty:
        sub = ngs_ru[(ngs_ru["season"] == year)]
        for t in teams:
            g = sub[sub["team_abbr"] == t].copy()
            if g.empty:
                continue
            agg = g.groupby(["player_id"], as_index=False).agg({
                "rush_yards_over_expected_per_att": "mean",
                "efficiency": "mean",
                "percent_attempts_gte_eight_defenders": "mean",
            })
            z_ryoe = _zscore(agg["rush_yards_over_expected_per_att"])
            z_eff  = _zscore(agg["efficiency"])
            z_8box = _zscore(agg["percent_attempts_gte_eight_defenders"])

            carry_raw = 0.45*z_ryoe + 0.25*z_eff - 0.20*z_8box
            for i, r in agg.iterrows():
                pid = r["player_id"]
                rush_carry[t][pid] = float(np.clip(np.exp(carry_raw.iloc[i]), 0.5, 1.5))

    return {"rec_target": rec_target, "rec_complete": rec_complete, "rush_carry": rush_carry}