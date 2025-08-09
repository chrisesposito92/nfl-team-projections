from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .config import (
    SIM_PHI_TARGET,
    SIM_PHI_RUSH,
    SIM_SIGMA_FRAC,
    RECEIVING_POS,
    RUSHING_POS,
    RESCALE_EPS,
    SIGMA_LOOKBACK_SEASONS, TD_NB_ALPHA_BASE, SIGMA_FLOOR, SIGMA_CEIL
)
from .data import load_depth_charts, load_team_scoring_sigma
from .utils import allocate_total_by_weights
from functools import lru_cache

@lru_cache(maxsize=8)
def _sigma_profile_cached(year: int):
    return load_team_scoring_sigma(year, SIGMA_LOOKBACK_SEASONS, floor=SIGMA_FLOOR, ceil=SIGMA_CEIL)

def _nb_td_draw(rng: np.random.Generator, mean_tds: float, sigma_mult: float) -> int:
    """
    Negative-binomial draw via Gamma-Poisson mixture with per-team sigma multiplier.
    Larger sigma_mult -> more variance (smaller effective alpha).
    """
    m = float(max(0.0, mean_tds))
    if m <= 0.0:
        return 0
    alpha_eff = float(TD_NB_ALPHA_BASE / max(1e-6, sigma_mult))
    if alpha_eff <= 1e-6:
        alpha_eff = 1e-6
    lam = rng.gamma(shape=alpha_eff, scale=m / alpha_eff)
    return int(rng.poisson(lam))

def _dirichlet_sample(base_shares: np.ndarray, phi: float, rng: np.random.RandomState) -> np.ndarray:
    base = np.asarray(base_shares, dtype=float)
    if base.size == 0:
        return base
    s = base.sum()
    if not np.isfinite(s) or s <= 0:
        base = np.ones_like(base) / len(base)
    else:
        base = base / s
    alpha = np.maximum(base * float(phi), 1e-6)
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

    for col in ["catch_rate", "ypt", "rec_td_per_target", "yprush", "rush_td_per_att"]:
        pos_mean = x.groupby("position")[col].transform("mean")
        x[col] = x[col].fillna(pos_mean).fillna(0.0)

    x["catch_rate"] = np.clip(x["catch_rate"].astype(float), 0.0, 1.0)
    x["ypt"] = np.maximum(x["ypt"].astype(float), 0.0)
    x["rec_td_per_target"] = np.clip(x["rec_td_per_target"].astype(float), 0.0, 1.0)
    x["yprush"] = np.maximum(x["yprush"].astype(float), 0.0)
    x["rush_td_per_att"] = np.clip(x["rush_td_per_att"].astype(float), 0.0, 1.0)

    return x[["player_id", "catch_rate", "ypt", "rec_td_per_target", "yprush", "rush_td_per_att"]]


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


def _safe_rescale_1d(x: np.ndarray, target: float) -> np.ndarray:
    if x.size == 0:
        return x
    s = float(np.nansum(x))
    if not np.isfinite(target) or target <= RESCALE_EPS or s <= RESCALE_EPS:
        return np.zeros_like(x, dtype=float)
    return (x.astype(float) * (float(target) / s))


def simulate_from_point(
    point_df: pd.DataFrame,
    team_pred: Dict[str, float],
    team: str,
    year: int,
    week: int | None = None,
    draws: int | None = None,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monte Carlo from point estimates with:
      - Dirichlet allocation for targets/rush attempts
      - Red-zone TD tilt via team φ (phi) for pass/rush TDs
      - TD overdispersion via team σ (sigma) using Gamma-Poisson (NegBin)
      - Numerically safe ops (no divide-by-zero warnings)
    Returns (team_summary, player_summary).
    """
    # Lazy imports for defaults & constants to avoid import-time NameError
    from .config import (
        SIM_DEFAULT_DRAWS,
        SIM_DEFAULT_SEED,
        SIM_PHI_TARGET,
        SIM_PHI_RUSH,
        PHI_PASS_MIN,
        PHI_PASS_MAX,
        PHI_RUSH_MIN,
        PHI_RUSH_MAX,
    )
    from .adjust import _rz_profile_cached

    # Resolve defaults
    if draws is None:
        draws = SIM_DEFAULT_DRAWS
    if seed is None:
        seed = SIM_DEFAULT_SEED

    df = point_df.copy()
    rng = np.random.RandomState(int(seed))

    # Merge per-player efficiency computed from the point estimates
    eff = _compute_eff_from_point(df)
    df = df.merge(eff, on="player_id", how="left").fillna(0.0)

    # Masks / indices
    rx_mask = df["position"].isin(RECEIVING_POS).to_numpy()
    ru_mask = df["position"].isin(RUSHING_POS).to_numpy()
    rx_idx = np.where(rx_mask)[0]
    ru_idx = np.where(ru_mask)[0]

    # Base shares (fallback to zeros if missing)
    base_rx = df.loc[rx_mask, "target_share"].to_numpy(dtype=float) if "target_share" in df.columns else np.zeros(rx_idx.size, dtype=float)
    base_ru = df.loc[ru_mask, "rush_attempt_share"].to_numpy(dtype=float) if "rush_attempt_share" in df.columns else np.zeros(ru_idx.size, dtype=float)

    # Red-zone shares (try multiple common names; fallback to zeros)
    def _get_first_available(mask: np.ndarray, cols: list[str]) -> np.ndarray:
        for c in cols:
            if c in df.columns:
                arr = df.loc[mask, c].to_numpy(dtype=float)
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.zeros(mask.sum(), dtype=float)

    rz_rx_share = _get_first_available(rx_mask, ["rz_target_share", "target_share_rz", "rz_tgt_share"])
    rz_ru_share = _get_first_available(ru_mask, ["rz_rush_share", "rush_attempt_share_rz", "rz_att_share"])

    # QB1 index for passing attribution
    qb1_idx = _qb1_index(df, team, year)

    # Receiving efficiencies (safe, clipped)
    rx_cr = df.loc[rx_mask, "catch_rate"].to_numpy(dtype=float) if "catch_rate" in df.columns else np.zeros(rx_idx.size, dtype=float)
    rx_cr = np.clip(np.nan_to_num(rx_cr, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    rx_ypt = df.loc[rx_mask, "ypt"].to_numpy(dtype=float) if "ypt" in df.columns else np.zeros(rx_idx.size, dtype=float)
    rx_ypt = np.maximum(np.nan_to_num(rx_ypt, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    # Safe division: ypr = ypt / cr  (0 if cr == 0)
    rx_ypr = np.divide(rx_ypt, rx_cr, out=np.zeros_like(rx_ypt), where=(rx_cr > 0))

    rx_rrate = df.loc[rx_mask, "rec_td_per_target"].to_numpy(dtype=float) if "rec_td_per_target" in df.columns else np.zeros(rx_idx.size, dtype=float)
    rx_rrate = np.clip(np.nan_to_num(rx_rrate, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    # Rushing efficiencies (safe, clipped)
    ru_yprush = df.loc[ru_mask, "yprush"].to_numpy(dtype=float) if "yprush" in df.columns else np.zeros(ru_idx.size, dtype=float)
    ru_yprush = np.maximum(np.nan_to_num(ru_yprush, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    ru_rtrate = df.loc[ru_mask, "rush_td_per_att"].to_numpy(dtype=float) if "rush_td_per_att" in df.columns else np.zeros(ru_idx.size, dtype=float)
    ru_rtrate = np.clip(np.nan_to_num(ru_rtrate, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    # Team-level φ (red-zone tilt) and σ (TD overdispersion)
    try:
        rz_prof = _rz_profile_cached(year)
        if team in rz_prof.index:
            phi_pass = float(rz_prof.loc[team, "phi_pass_td_rz"])
            phi_rush = float(rz_prof.loc[team, "phi_rush_td_rz"])
        else:
            phi_pass = 1.0
            phi_rush = 1.0
    except Exception:
        phi_pass = 1.0
        phi_rush = 1.0

    phi_pass = float(np.clip(phi_pass, PHI_PASS_MIN, PHI_PASS_MAX))
    phi_rush = float(np.clip(phi_rush, PHI_RUSH_MIN, PHI_RUSH_MAX))

    try:
        s_prof = _sigma_profile_cached(year)
        sigma_td = float(s_prof.loc[team, "sigma_td"]) if team in s_prof.index else 1.0
    except Exception:
        sigma_td = 1.0
    sigma_td = float(max(0.0, sigma_td))

    # Helper: sample TD totals with overdispersion (Gamma-Poisson => NegBin)
    def _sample_tds_nb(mean_td: float) -> float:
        m = max(0.0, float(mean_td))
        if m == 0.0:
            return 0.0
        if sigma_td <= 1e-9:
            return float(rng.poisson(m))
        r = 1.0 / max(1e-9, sigma_td)  # larger sigma => smaller r => more variance
        lam = rng.gamma(shape=r, scale=m / r)
        return float(rng.poisson(max(0.0, lam)))

    # Stats to aggregate
    stats = [
        "targets",
        "receptions",
        "rec_yards",
        "rec_tds",
        "rush_att",
        "rush_yards",
        "rush_tds",
        "pass_yards",
        "pass_tds",
    ]
    agg = {s: [] for s in stats}
    team_agg = {k: [] for k in ["pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds"]}

    for _ in range(int(draws)):
        t = _sample_team_totals(team_pred, rng)

        # Replace TD totals with σ-calibrated draws (keeps means at team_pred)
        t["pass_tds"] = _sample_tds_nb(team_pred.get("pass_tds", t.get("pass_tds", 0.0)))
        t["rush_tds"] = _sample_tds_nb(team_pred.get("rush_tds", t.get("rush_tds", 0.0)))

        for k in team_agg.keys():
            team_agg[k].append(t.get(k, 0.0))

        draw = {s: np.zeros(len(df), dtype=float) for s in stats}

        # Receiving draw
        if rx_idx.size > 0:
            rx_share = _dirichlet_sample(base_rx, SIM_PHI_TARGET, rng)
            rx_targets = max(0.0, t["pass_attempts"]) * rx_share
            draw["targets"][rx_idx] = rx_targets

            n_binom = np.maximum(0, np.rint(rx_targets)).astype(np.int64)
            recs = rng.binomial(n=n_binom, p=rx_cr)
            draw["receptions"][rx_idx] = recs
            draw["rec_yards"][rx_idx] = recs * rx_ypr

            # TD allocation with red-zone tilt
            base_w_rec = rx_targets * rx_rrate
            rz_w_rec = (max(0.0, t["pass_attempts"]) * rz_rx_share) * rx_rrate
            weights_rec = np.nan_to_num(base_w_rec + (phi_pass - 1.0) * rz_w_rec, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(weights_rec.sum()) or weights_rec.sum() <= 0:
                alloc_rec_tds = np.zeros_like(rx_targets)
            else:
                alloc_rec_tds = allocate_total_by_weights(t["pass_tds"], np.maximum(0.0, weights_rec))
            draw["rec_tds"][rx_idx] = alloc_rec_tds

        # Rushing draw
        if ru_idx.size > 0:
            ru_share = _dirichlet_sample(base_ru, SIM_PHI_RUSH, rng)
            ru_att = max(0.0, t["rush_attempts"]) * ru_share
            draw["rush_att"][ru_idx] = ru_att
            draw["rush_yards"][ru_idx] = ru_att * ru_yprush

            # TD allocation with red-zone tilt
            base_w_rush = ru_att * ru_rtrate
            rz_w_rush = (max(0.0, t["rush_attempts"]) * rz_ru_share) * ru_rtrate
            weights_rush = np.nan_to_num(base_w_rush + (phi_rush - 1.0) * rz_w_rush, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(weights_rush.sum()) or weights_rush.sum() <= 0:
                alloc_rush_tds = np.zeros_like(ru_att)
            else:
                alloc_rush_tds = allocate_total_by_weights(t["rush_tds"], np.maximum(0.0, weights_rush))
            draw["rush_tds"][ru_idx] = alloc_rush_tds

        # QB attribution for passing yards/TDs
        qb_pass_yards = np.zeros(len(df), dtype=float)
        qb_pass_tds = np.zeros(len(df), dtype=float)
        if qb1_idx.size > 0:
            qb_pass_yards[qb1_idx[0]] = max(0.0, t["pass_yards"])
            qb_pass_tds[qb1_idx[0]] = max(0.0, t["pass_tds"])
        draw["pass_yards"] = qb_pass_yards
        draw["pass_tds"] = qb_pass_tds

        # Team-level reconciliation to totals
        if rx_idx.size > 0:
            draw["rec_yards"][rx_idx] = _safe_rescale_1d(draw["rec_yards"][rx_idx], max(0.0, t["pass_yards"]))
            draw["rec_tds"][rx_idx] = _safe_rescale_1d(draw["rec_tds"][rx_idx], max(0.0, t["pass_tds"]))
        if ru_idx.size > 0:
            draw["rush_yards"][ru_idx] = _safe_rescale_1d(draw["rush_yards"][ru_idx], max(0.0, t["rush_yards"]))
            draw["rush_tds"][ru_idx] = _safe_rescale_1d(draw["rush_tds"][ru_idx], max(0.0, t["rush_tds"]))

        for s in stats:
            agg[s].append(draw[s])

    # Aggregate players
    out = {}
    for s in stats:
        mat = np.stack(agg[s], axis=0) if len(agg[s]) > 0 else np.zeros((0, len(df)), dtype=float)
        out[f"{s}_mean"] = mat.mean(axis=0) if mat.size else np.zeros(len(df), dtype=float)
        out[f"{s}_p10"] = np.percentile(mat, 10, axis=0) if mat.size else np.zeros(len(df), dtype=float)
        out[f"{s}_p50"] = np.percentile(mat, 50, axis=0) if mat.size else np.zeros(len(df), dtype=float)
        out[f"{s}_p90"] = np.percentile(mat, 90, axis=0) if mat.size else np.zeros(len(df), dtype=float)

    player_summary = df[["player_id", "player_name", "position"]].copy()
    for k, v in out.items():
        player_summary[k] = v
    player_summary = player_summary.sort_values(["position", "player_name"]).reset_index(drop=True)

    # Aggregate team
    team_out = {}
    for k, arr in team_agg.items():
        a = np.array(arr, dtype=float)
        team_out[f"{k}_mean"] = float(a.mean()) if a.size else 0.0
        team_out[f"{k}_p10"] = float(np.percentile(a, 10)) if a.size else 0.0
        team_out[f"{k}_p50"] = float(np.percentile(a, 50)) if a.size else 0.0
        team_out[f"{k}_p90"] = float(np.percentile(a, 90)) if a.size else 0.0

    team_summary = pd.DataFrame([team_out])

    return team_summary, player_summary