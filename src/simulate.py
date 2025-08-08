from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .config import SIM_PHI_TARGET, SIM_PHI_RUSH, SIM_SIGMA_FRAC, RECEIVING_POS, RUSHING_POS
from .data import load_depth_charts
from .utils import allocate_total_by_weights

def _dirichlet_sample(base_shares: np.ndarray, phi: float, rng: np.random.RandomState) -> np.ndarray:
    base = base_shares.astype(float)
    if base.sum() <= 0:
        base = np.ones_like(base) / len(base)
    else:
        base = base / base.sum()
    alpha = np.maximum(base * phi, 1e-6)
    return rng.dirichlet(alpha)

def _sample_team_totals(mu: Dict[str, float], rng: np.random.RandomState) -> Dict[str, float]:
    out = {}
    out["pass_attempts"] = max(0.0, rng.normal(mu["pass_attempts"], SIM_SIGMA_FRAC["pass_attempts"] * max(mu["pass_attempts"], 1.0)))
    out["rush_attempts"] = max(0.0, rng.normal(mu["rush_attempts"], SIM_SIGMA_FRAC["rush_attempts"] * max(mu["rush_attempts"], 1.0)))
    out["pass_yards"] = max(0.0, rng.normal(mu["pass_yards"], SIM_SIGMA_FRAC["pass_yards"] * max(mu["pass_yards"], 1.0)))
    out["rush_yards"] = max(0.0, rng.normal(mu["rush_yards"], SIM_SIGMA_FRAC["rush_yards"] * max(mu["rush_yards"], 1.0)))
    out["pass_tds"] = float(rng.poisson(max(mu["pass_tds"], 0.0)))
    out["rush_tds"] = float(rng.poisson(max(mu["rush_tds"], 0.0)))
    return out

def _compute_eff_from_point(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["catch_rate"] = np.where(x["proj_targets"] > 0, x["proj_receptions"] / x["proj_targets"], np.nan)
    x["ypt"] = np.where(x["proj_targets"] > 0, x["proj_rec_yards"] / x["proj_targets"], np.nan)
    x["rec_td_per_target"] = np.where(x["proj_targets"] > 0, x["proj_rec_tds"] / x["proj_targets"], np.nan)
    x["yprush"] = np.where(x["proj_rush_att"] > 0, x["proj_rush_yards"] / x["proj_rush_att"], np.nan)
    x["rush_td_per_att"] = np.where(x["proj_rush_att"] > 0, x["proj_rush_tds"] / x["proj_rush_att"], np.nan)
    for col in ["catch_rate","ypt","rec_td_per_target","yprush","rush_td_per_att"]:
        pos_mean = x.groupby("position")[col].transform("mean")
        x[col] = x[col].fillna(pos_mean).fillna(0.0)
    return x[["player_id","catch_rate","ypt","rec_td_per_target","yprush","rush_td_per_att"]]

def _qb1_index(df: pd.DataFrame, team: str, year: int) -> np.ndarray:
    qb_idx = df.index[df["position"] == "QB"].values
    if qb_idx.size == 0:
        return qb_idx
    try:
        dc = load_depth_charts([year])
        dc = dc[dc["team"] == team].copy()
        dc["dt"] = pd.to_datetime(dc["dt"], errors="coerce")
        dc = dc.sort_values("dt")
        qb1 = dc[(dc["pos_abb"] == "QB") & (pd.to_numeric(dc["pos_rank"], errors="coerce") == 1)]
        if not qb1.empty:
            name = qb1.iloc[-1]["player_name"]
            match = df.index[(df["position"] == "QB") & (df["player_name"] == name)].values
            if match.size > 0:
                return match
    except Exception:
        pass
    return np.array([qb_idx[0]])

def simulate_from_point(point_df: pd.DataFrame, team_pred: Dict[str, float], team: str, year: int, n_draws: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = point_df.copy()
    rng = np.random.RandomState(seed)
    eff = _compute_eff_from_point(df)
    df = df.merge(eff, on="player_id", how="left").fillna(0.0)

    rx_mask = df["position"].isin(RECEIVING_POS)
    ru_mask = df["position"].isin(RUSHING_POS)
    rx_idx = df.index[rx_mask].values
    ru_idx = df.index[ru_mask].values
    base_rx = df.loc[rx_mask, "target_share"].values if "target_share" in df.columns else np.zeros(rx_idx.size)
    base_ru = df.loc[ru_mask, "rush_attempt_share"].values if "rush_attempt_share" in df.columns else np.zeros(ru_idx.size)
    qb1_idx = _qb1_index(df, team, year)

    stats = ["targets","receptions","rec_yards","rec_tds","rush_att","rush_yards","rush_tds","pass_yards","pass_tds"]
    agg = {s: [] for s in stats}
    team_agg = {k: [] for k in ["pass_attempts","pass_yards","pass_tds","rush_attempts","rush_yards","rush_tds"]}

    for _ in range(n_draws):
        t = _sample_team_totals(team_pred, rng)
        for k in team_agg.keys():
            team_agg[k].append(t[k])

        draw = {s: np.zeros(len(df), dtype=float) for s in stats}

        if rx_idx.size > 0:
            rx_share = _dirichlet_sample(base_rx, SIM_PHI_TARGET, rng)
            rx_targets = t["pass_attempts"] * rx_share
            draw["targets"][rx_idx] = rx_targets
            cr = df.loc[rx_idx, "catch_rate"].values
            ypt = df.loc[rx_idx, "ypt"].values
            ypr = np.where(cr > 0, ypt / cr, 0.0)
            n_binom = np.maximum(0, np.rint(rx_targets)).astype(int)
            recs = rng.binomial(n=n_binom, p=np.clip(cr, 0.0, 1.0))
            draw["receptions"][rx_idx] = recs
            draw["rec_yards"][rx_idx] = recs * ypr
            rrate = df.loc[rx_idx, "rec_td_per_target"].values
            weights_rec = rx_targets * rrate
            draw["rec_tds"][rx_idx] = allocate_total_by_weights(t["pass_tds"], weights_rec)

        if ru_idx.size > 0:
            ru_share = _dirichlet_sample(base_ru, SIM_PHI_RUSH, rng)
            ru_att = t["rush_attempts"] * ru_share
            draw["rush_att"][ru_idx] = ru_att
            yprush = df.loc[ru_idx, "yprush"].values
            draw["rush_yards"][ru_idx] = ru_att * yprush
            rtrate = df.loc[ru_idx, "rush_td_per_att"].values
            weights_rush = ru_att * rtrate
            draw["rush_tds"][ru_idx] = allocate_total_by_weights(t["rush_tds"], weights_rush)

        qb_pass_yards = np.zeros(len(df), dtype=float)
        qb_pass_tds = np.zeros(len(df), dtype=float)
        if qb1_idx.size > 0:
            qb_pass_yards[qb1_idx[0]] = t["pass_yards"]
            qb_pass_tds[qb1_idx[0]] = t["pass_tds"]
        draw["pass_yards"] = qb_pass_yards
        draw["pass_tds"] = qb_pass_tds

        for s in stats:
            agg[s].append(draw[s])

    out = {}
    for s in stats:
        mat = np.stack(agg[s], axis=0)
        out[f"{s}_mean"] = mat.mean(axis=0)
        out[f"{s}_p10"] = np.percentile(mat, 10, axis=0)
        out[f"{s}_p50"] = np.percentile(mat, 50, axis=0)
        out[f"{s}_p90"] = np.percentile(mat, 90, axis=0)

    sim_players = df[["player_id","player_name","position"]].copy()
    for k, v in out.items():
        sim_players[k] = v
    sim_players = sim_players.sort_values(["position","player_name"]).reset_index(drop=True)

    team_out = {}
    for k, arr in team_agg.items():
        a = np.array(arr, dtype=float)
        team_out[f"{k}_mean"] = float(a.mean())
        team_out[f"{k}_p10"] = float(np.percentile(a, 10))
        team_out[f"{k}_p50"] = float(np.percentile(a, 50))
        team_out[f"{k}_p90"] = float(np.percentile(a, 90))
    sim_team = pd.DataFrame([team_out])

    return sim_players, sim_team