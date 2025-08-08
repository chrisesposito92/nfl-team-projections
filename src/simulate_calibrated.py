from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .config import (
    SIM_PHI_TARGET,
    SIM_PHI_RUSH,
    SIM_SIGMA_FRAC,
    SIM_DEFAULT_RHO,
    SIM_NB_THETA,
    SIM_CALIBRATION_FILE,
    RECEIVING_POS,
    RUSHING_POS,
)
from .simulate import _compute_eff_from_point, _dirichlet_sample, _qb1_index
from .utils import allocate_total_by_weights

def _load_calibration() -> Dict[str, object]:
    if os.path.exists(SIM_CALIBRATION_FILE):
        with open(SIM_CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return {
        "sigma_frac": SIM_SIGMA_FRAC,
        "rho": SIM_DEFAULT_RHO,
        "nb_theta": SIM_NB_THETA,
    }

def _mv_sample_team_totals(mu: Dict[str, float], calib: Dict[str, object], rng: np.random.RandomState) -> Dict[str, float]:
    order = ["pass_attempts", "rush_attempts", "pass_yards", "rush_yards"]
    m = np.array([float(mu[k]) for k in order], dtype=float)

    sigma_frac = calib.get("sigma_frac", SIM_SIGMA_FRAC)
    s_abs = np.array([
        sigma_frac["pass_attempts"] * max(mu["pass_attempts"], 1.0),
        sigma_frac["rush_attempts"] * max(mu["rush_attempts"], 1.0),
        sigma_frac["pass_yards"] * max(mu["pass_yards"], 1.0),
        sigma_frac["rush_yards"] * max(mu["rush_yards"], 1.0),
    ], dtype=float)

    rho = np.array(calib.get("rho", SIM_DEFAULT_RHO), dtype=float)
    D = np.diag(s_abs)
    cov = D @ rho @ D

    draw = rng.multivariate_normal(mean=m, cov=cov)
    draw = np.maximum(draw, 0.0)

    nb_theta = calib.get("nb_theta", SIM_NB_THETA)
    def _nb(mean, theta):
        mean = max(float(mean), 0.0)
        theta = float(theta)
        if theta <= 0:
            return float(rng.poisson(mean))
        shape = theta
        scale = mean / theta
        lam = rng.gamma(shape, scale)
        return float(rng.poisson(lam))

    out = {
        "pass_attempts": float(draw[0]),
        "rush_attempts": float(draw[1]),
        "pass_yards": float(draw[2]),
        "rush_yards": float(draw[3]),
        "pass_tds": _nb(mu["pass_tds"], nb_theta.get("pass_tds", 0.0)),
        "rush_tds": _nb(mu["rush_tds"], nb_theta.get("rush_tds", 0.0)),
    }
    return out

def simulate_from_point_calibrated(point_df: pd.DataFrame, team_pred: Dict[str, float], team: str, year: int, n_draws: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = point_df.copy()
    rng = np.random.RandomState(seed)
    calib = _load_calibration()

    eff = _compute_eff_from_point(df)
    df = df.merge(eff, on="player_id", how="left").fillna(0.0)

    rx_mask = df["position"].isin(RECEIVING_POS).values
    ru_mask = df["position"].isin(RUSHING_POS).values
    rx_idx = np.where(rx_mask)[0]
    ru_idx = np.where(ru_mask)[0]

    base_rx = df.loc[rx_mask, "target_share"].to_numpy(dtype=float) if "target_share" in df.columns else np.zeros(rx_idx.size)
    base_ru = df.loc[ru_mask, "rush_attempt_share"].to_numpy(dtype=float) if "rush_attempt_share" in df.columns else np.zeros(ru_idx.size)

    qb1_idx = _qb1_index(df, team, year)

    rx_cr = np.clip(np.nan_to_num(df.loc[rx_mask, "catch_rate"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    rx_ypt = np.maximum(np.nan_to_num(df.loc[rx_mask, "ypt"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    rx_ypr = np.divide(rx_ypt, rx_cr, out=np.zeros_like(rx_ypt), where=(rx_cr > 0))
    rx_rrate = np.clip(np.nan_to_num(df.loc[rx_mask, "rec_td_per_target"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    ru_yprush = np.maximum(np.nan_to_num(df.loc[ru_mask, "yprush"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    ru_rtrate = np.clip(np.nan_to_num(df.loc[ru_mask, "rush_td_per_att"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    stats = ["targets","receptions","rec_yards","rec_tds","rush_att","rush_yards","rush_tds","pass_yards","pass_tds"]
    agg = {s: [] for s in stats}
    team_agg = {k: [] for k in ["pass_attempts","pass_yards","pass_tds","rush_attempts","rush_yards","rush_tds"]}

    for _ in range(int(n_draws)):
        t = _mv_sample_team_totals(team_pred, calib, rng)
        for k in team_agg.keys():
            team_agg[k].append(t[k])

        draw = {s: np.zeros(len(df), dtype=float) for s in stats}

        if rx_idx.size > 0:
            rx_share = _dirichlet_sample(base_rx, SIM_PHI_TARGET, rng)
            rx_targets = t["pass_attempts"] * rx_share
            draw["targets"][rx_idx] = rx_targets

            n_binom = np.maximum(0, np.rint(rx_targets)).astype(np.int64)
            recs = rng.binomial(n=n_binom, p=rx_cr)
            draw["receptions"][rx_idx] = recs
            draw["rec_yards"][rx_idx] = recs * rx_ypr

            weights_rec = rx_targets * rx_rrate
            if not np.isfinite(weights_rec.sum()) or weights_rec.sum() <= 0:
                alloc_rec_tds = np.zeros_like(rx_targets)
            else:
                alloc_rec_tds = allocate_total_by_weights(t["pass_tds"], weights_rec)
            draw["rec_tds"][rx_idx] = alloc_rec_tds

        if ru_idx.size > 0:
            ru_share = _dirichlet_sample(base_ru, SIM_PHI_RUSH, rng)
            ru_att = t["rush_attempts"] * ru_share
            draw["rush_att"][ru_idx] = ru_att
            draw["rush_yards"][ru_idx] = ru_att * ru_yprush

            weights_rush = ru_att * ru_rtrate
            if not np.isfinite(weights_rush.sum()) or weights_rush.sum() <= 0:
                alloc_rush_tds = np.zeros_like(ru_att)
            else:
                alloc_rush_tds = allocate_total_by_weights(t["rush_tds"], weights_rush)
            draw["rush_tds"][ru_idx] = alloc_rush_tds

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