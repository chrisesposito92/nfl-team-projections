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
    Monte Carlo from point estimates with FULL efficiency sampling:
      - Dirichlet allocation for targets/rush attempts
      - Logit-normal catch rate sampling (Beta-like without SciPy)
      - Lognormal YPT (=> YPR) sampling for receivers
      - Lognormal YPC sampling for rushers
      - Correlation between CR and YPT (aDOT-like), exposure-based shrink
      - Team-level TD overdispersion (Gamma-Poisson) and red-zone φ tilt
      - Sum-to-team reconciliation per draw

    Returns (team_summary, player_summary).
    """
    # Lazy imports for defaults & constants (avoid import-time failures)
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

    # Optional efficiency sampling knobs (fallbacks if not in config.py)
    try:
        from .config import (
            EFF_RX_LOGIT_SD,          # baseline logit sd for CR
            EFF_YPT_CV,               # baseline CV for YPT
            EFF_RUSH_YPC_CV,          # baseline CV for rush YPC
            EFF_EXPOSURE_SHRINK,      # larger => stronger shrink with volume
            EFF_RHO_RX,               # corr(CR, YPT) via Gaussian copula (negative)
            EFF_YPR_MAX,              # hard cap on yards/reception to prevent wild tails
            EFF_RUSH_YPC_MAX,         # hard cap on rush YPC
        )
    except Exception:
        # Sensible defaults if those keys are not present in config.py
        EFF_RX_LOGIT_SD = 0.35
        EFF_YPT_CV = 0.35
        EFF_RUSH_YPC_CV = 0.25
        EFF_EXPOSURE_SHRINK = 8.0
        EFF_RHO_RX = -0.35
        EFF_YPR_MAX = 35.0
        EFF_RUSH_YPC_MAX = 9.0

    # Resolve defaults
    if draws is None:
        draws = SIM_DEFAULT_DRAWS
    if seed is None:
        seed = SIM_DEFAULT_SEED

    df = point_df.copy()
    rng = np.random.RandomState(int(seed))

    # Merge per-player efficiency computed from the point estimates (safe + clipped)
    eff = _compute_eff_from_point(df)
    df = df.merge(eff, on="player_id", how="left").fillna(0.0)

    # Masks / indices
    rx_mask = df["position"].isin(RECEIVING_POS).to_numpy()
    ru_mask = df["position"].isin(RUSHING_POS).to_numpy()
    rx_idx = np.where(rx_mask)[0]
    ru_idx = np.where(ru_mask)[0]

    # Base shares (fallback to zeros if missing)
    base_rx = (
        df.loc[rx_mask, "target_share"].to_numpy(dtype=float)
        if "target_share" in df.columns else np.zeros(rx_idx.size, dtype=float)
    )
    base_ru = (
        df.loc[ru_mask, "rush_attempt_share"].to_numpy(dtype=float)
        if "rush_attempt_share" in df.columns else np.zeros(ru_idx.size, dtype=float)
    )

    # Helper to grab first available RZ share column
    def _get_first_available(mask: np.ndarray, cols: list[str]) -> np.ndarray:
        for c in cols:
            if c in df.columns:
                arr = df.loc[mask, c].to_numpy(dtype=float)
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.zeros(mask.sum(), dtype=float)

    # Red-zone shares (if present)
    rz_rx_share = _get_first_available(rx_mask, ["rz_target_share", "target_share_rz", "rz_tgt_share"])
    rz_ru_share = _get_first_available(ru_mask, ["rz_rush_share", "rush_attempt_share_rz", "rz_att_share"])

    # QB1 index for passing attribution
    qb1_idx = _qb1_index(df, team, year)

    # Receiving base efficiencies (safe, clipped)
    rx_cr_base = (
        df.loc[rx_mask, "catch_rate"].to_numpy(dtype=float)
        if "catch_rate" in df.columns else np.zeros(rx_idx.size, dtype=float)
    )
    rx_cr_base = np.clip(np.nan_to_num(rx_cr_base, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    rx_ypt_base = (
        df.loc[rx_mask, "ypt"].to_numpy(dtype=float)
        if "ypt" in df.columns else np.zeros(rx_idx.size, dtype=float)
    )
    rx_ypt_base = np.maximum(np.nan_to_num(rx_ypt_base, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    rx_rrate = (
        df.loc[rx_mask, "rec_td_per_target"].to_numpy(dtype=float)
        if "rec_td_per_target" in df.columns else np.zeros(rx_idx.size, dtype=float)
    )
    rx_rrate = np.clip(np.nan_to_num(rx_rrate, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    # Rushing base efficiencies (safe, clipped)
    ru_yprush_base = (
        df.loc[ru_mask, "yprush"].to_numpy(dtype=float)
        if "yprush" in df.columns else np.zeros(ru_idx.size, dtype=float)
    )
    ru_yprush_base = np.maximum(np.nan_to_num(ru_yprush_base, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    ru_rtrate = (
        df.loc[ru_mask, "rush_td_per_att"].to_numpy(dtype=float)
        if "rush_td_per_att" in df.columns else np.zeros(ru_idx.size, dtype=float)
    )
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

    # Helper: sample TD totals with overdispersion (Gamma-Poisson => NegBin-like)
    def _sample_tds_nb(mean_td: float) -> float:
        m = max(0.0, float(mean_td))
        if m == 0.0:
            return 0.0
        if sigma_td <= 1e-9:
            return float(rng.poisson(m))
        r = 1.0 / max(1e-9, sigma_td)  # larger sigma => smaller r => more variance
        lam = rng.gamma(shape=r, scale=m / r)
        return float(rng.poisson(max(0.0, lam)))

    # Safe helpers for transforms
    def _inv_logit(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    # Stats to aggregate
    stats = [
        "targets", "receptions", "rec_yards", "rec_tds",
        "rush_att", "rush_yards", "rush_tds",
        "pass_yards", "pass_tds",
    ]
    agg = {s: [] for s in stats}
    team_agg = {k: [] for k in ["pass_attempts", "pass_yards", "pass_tds", "rush_attempts", "rush_yards", "rush_tds"]}

    for _ in range(int(draws)):
        # Team totals (incl. yardage normals & TD NegBin draw)
        t = _sample_team_totals(team_pred, rng)
        t["pass_tds"] = _sample_tds_nb(team_pred.get("pass_tds", t.get("pass_tds", 0.0)))
        t["rush_tds"] = _sample_tds_nb(team_pred.get("rush_tds", t.get("rush_tds", 0.0)))

        for k in team_agg.keys():
            team_agg[k].append(t.get(k, 0.0))

        # Per-draw containers
        draw = {s: np.zeros(len(df), dtype=float) for s in stats}

        # -------- Receiving draw (shares + efficiency sampling) --------
        if rx_idx.size > 0:
            # Shares -> targets
            rx_share = _dirichlet_sample(base_rx, SIM_PHI_TARGET, rng)
            rx_targets = max(0.0, t["pass_attempts"]) * rx_share
            draw["targets"][rx_idx] = rx_targets

            # Exposure-based shrink factors
            exp_shrink = np.sqrt(1.0 + rx_targets / max(EFF_EXPOSURE_SHRINK, 1e-6))
            s_cr = EFF_RX_LOGIT_SD / exp_shrink  # logit sd for CR
            cv_ypt = EFF_YPT_CV / exp_shrink     # CV for YPT
            sigma_ypt = np.sqrt(np.log1p(np.square(np.clip(cv_ypt, 0.0, 10.0))))

            # Correlated shocks per player
            z_common = rng.normal(size=rx_idx.size)
            eps1 = rng.normal(size=rx_idx.size)
            eps2 = rng.normal(size=rx_idx.size)
            rho = float(np.clip(EFF_RHO_RX, -0.95, 0.95))
            z_cr = rho * z_common + np.sqrt(max(0.0, 1.0 - rho * rho)) * eps1
            z_ypt = (-rho) * z_common + np.sqrt(max(0.0, 1.0 - rho * rho)) * eps2

            # Catch rate ~ logit-normal around base mean (keep in (0,1))
            p0 = np.clip(rx_cr_base, 1e-4, 1.0 - 1e-4)
            mu_cr = np.log(p0 / (1.0 - p0))
            logit_cr_draw = mu_cr + s_cr * z_cr
            cr_draw = np.clip(_inv_logit(logit_cr_draw), 0.0, 1.0)

            # YPT ~ lognormal around base mean
            m_ypt = np.maximum(rx_ypt_base, 1e-3)
            mu_ypt = np.log(m_ypt) - 0.5 * (sigma_ypt ** 2)
            ln_ypt_draw = mu_ypt + sigma_ypt * z_ypt
            ypt_draw = np.exp(ln_ypt_draw)

            # Receptions
            n_binom = np.maximum(0, np.rint(rx_targets)).astype(np.int64)
            recs = rng.binomial(n=n_binom, p=np.clip(cr_draw, 0.0, 1.0))
            draw["receptions"][rx_idx] = recs

            # YPR = YPT / CR (safe), with a gentle hard cap to avoid absurd outliers
            small = 1e-6
            ypr_draw = np.divide(ypt_draw, np.maximum(cr_draw, small))
            ypr_draw = np.clip(ypr_draw, 0.0, float(EFF_YPR_MAX))

            # Receiving yards
            draw["rec_yards"][rx_idx] = recs * ypr_draw

            # TD allocation with red-zone tilt
            base_w_rec = rx_targets * rx_rrate
            rz_w_rec = (max(0.0, t["pass_attempts"]) * rz_rx_share) * rx_rrate
            weights_rec = np.nan_to_num(base_w_rec + (phi_pass - 1.0) * rz_w_rec, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(weights_rec.sum()) or weights_rec.sum() <= 0:
                alloc_rec_tds = np.zeros_like(rx_targets)
            else:
                alloc_rec_tds = allocate_total_by_weights(t["pass_tds"], np.maximum(0.0, weights_rec))
            draw["rec_tds"][rx_idx] = alloc_rec_tds

        # -------- Rushing draw (shares + YPC sampling) --------
        if ru_idx.size > 0:
            ru_share = _dirichlet_sample(base_ru, SIM_PHI_RUSH, rng)
            ru_att = max(0.0, t["rush_attempts"]) * ru_share
            draw["rush_att"][ru_idx] = ru_att

            # Exposure-based shrink for YPC
            exp_shrink_ru = np.sqrt(1.0 + ru_att / max(EFF_EXPOSURE_SHRINK, 1e-6))
            cv_yprush = EFF_RUSH_YPC_CV / exp_shrink_ru
            sigma_yprush = np.sqrt(np.log1p(np.square(np.clip(cv_yprush, 0.0, 10.0))))

            m_yprush = np.maximum(ru_yprush_base, 1e-3)
            mu_yprush = np.log(m_yprush) - 0.5 * (sigma_yprush ** 2)
            ln_yprush_draw = mu_yprush + sigma_yprush * rng.normal(size=ru_idx.size)
            yprush_draw = np.clip(np.exp(ln_yprush_draw), 0.0, float(EFF_RUSH_YPC_MAX))

            draw["rush_yards"][ru_idx] = ru_att * yprush_draw

            # TD allocation with red-zone tilt
            base_w_rush = ru_att * ru_rtrate
            rz_w_rush = (max(0.0, t["rush_attempts"]) * rz_ru_share) * ru_rtrate
            weights_rush = np.nan_to_num(base_w_rush + (phi_rush - 1.0) * rz_w_rush, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(weights_rush.sum()) or weights_rush.sum() <= 0:
                alloc_rush_tds = np.zeros_like(ru_att)
            else:
                alloc_rush_tds = allocate_total_by_weights(t["rush_tds"], np.maximum(0.0, weights_rush))
            draw["rush_tds"][ru_idx] = alloc_rush_tds

        # QB attribution for passing yards/TDs (all to QB1)
        qb_pass_yards = np.zeros(len(df), dtype=float)
        qb_pass_tds = np.zeros(len(df), dtype=float)
        if qb1_idx.size > 0:
            qb_pass_yards[qb1_idx[0]] = max(0.0, t["pass_yards"])
            qb_pass_tds[qb1_idx[0]] = max(0.0, t["pass_tds"])
        draw["pass_yards"] = qb_pass_yards
        draw["pass_tds"] = qb_pass_tds

        # Per-draw reconciliation to team totals (guards and preserves totals)
        if rx_idx.size > 0:
            draw["rec_yards"][rx_idx] = _safe_rescale_1d(draw["rec_yards"][rx_idx], max(0.0, t["pass_yards"]))
            draw["rec_tds"][rx_idx]   = _safe_rescale_1d(draw["rec_tds"][rx_idx],   max(0.0, t["pass_tds"]))
        if ru_idx.size > 0:
            draw["rush_yards"][ru_idx] = _safe_rescale_1d(draw["rush_yards"][ru_idx], max(0.0, t["rush_yards"]))
            draw["rush_tds"][ru_idx]   = _safe_rescale_1d(draw["rush_tds"][ru_idx],   max(0.0, t["rush_tds"]))

        for s in stats:
            agg[s].append(draw[s])

    # Aggregate players across draws
    out = {}
    for s in stats:
        mat = np.stack(agg[s], axis=0) if len(agg[s]) > 0 else np.zeros((0, len(df)), dtype=float)
        out[f"{s}_mean"] = mat.mean(axis=0) if mat.size else np.zeros(len(df), dtype=float)
        out[f"{s}_p10"]  = np.percentile(mat, 10, axis=0) if mat.size else np.zeros(len(df), dtype=float)
        out[f"{s}_p50"]  = np.percentile(mat, 50, axis=0) if mat.size else np.zeros(len(df), dtype=float)
        out[f"{s}_p90"]  = np.percentile(mat, 90, axis=0) if mat.size else np.zeros(len(df), dtype=float)

    player_summary = df[["player_id", "player_name", "position"]].copy()
    for k, v in out.items():
        player_summary[k] = v
    player_summary = player_summary.sort_values(["position", "player_name"]).reset_index(drop=True)

    # Aggregate team
    team_out = {}
    for k, arr in team_agg.items():
        a = np.array(arr, dtype=float)
        team_out[f"{k}_mean"] = float(a.mean()) if a.size else 0.0
        team_out[f"{k}_p10"]  = float(np.percentile(a, 10)) if a.size else 0.0
        team_out[f"{k}_p50"]  = float(np.percentile(a, 50)) if a.size else 0.0
        team_out[f"{k}_p90"]  = float(np.percentile(a, 90)) if a.size else 0.0

    team_summary = pd.DataFrame([team_out])
    return team_summary, player_summary