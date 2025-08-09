from __future__ import annotations
import numpy as np

from .config import (
    COMP_KAPPA, COMP_B_AIR, AIR_LN_SIGMA, YAC_LN_SIGMA,
    INT_RATE_BASE, INT_SLOPE_AIR, SACK_RATE_BASE, SACK_YARDS_MEAN, SACK_YARDS_SD,
    SCREEN_PROB_BASE, SCREEN_AIR_MEAN, SCREEN_AIR_SD,
    STUFF_RATE_BASE, RUN_LN_SIGMA, FUMBLE_RATE_RUN, FUMBLE_RATE_PASS
)
from .state import GameState
from .priors import TeamPriors

def _beta_from_mean_kappa(p: float, kappa: float) -> tuple[float, float]:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    a = p * kappa
    b = (1 - p) * kappa
    return a, b

def _lognormal_from_mean_sigma(mean: float, sigma: float) -> tuple[float, float]:
    """
    For X ~ LogNormal(mu, sigma): mean = exp(mu + 0.5*sigma^2)
    => mu = log(mean) - 0.5*sigma^2
    """
    m = float(max(mean, 1e-6))
    mu = np.log(m) - 0.5 * sigma * sigma
    return mu, sigma

def sim_pass(state: GameState, priors: TeamPriors, rng: np.random.RandomState) -> dict:
    out = {
        "play_type": "pass", "yards": 0.0, "complete": False, "sack": False,
        "interception": False, "fumble": False, "td": False, "turnover": False,
        "in_bounds": True, "time_elapsed": 0.0, "penalty_yards": 0.0,
        "air": 0.0, "yac": 0.0,
    }

    p_sack = float(np.clip(priors.sack_rate, 0.01, 0.15))
    if rng.rand() < p_sack:
        out["sack"] = True
        out["yards"] = float(rng.normal(SACK_YARDS_MEAN, SACK_YARDS_SD))
        out["time_elapsed"] = float(np.clip(rng.normal(6.0, 1.0), 4.0, 10.0))
        return out

    p_screen = float(np.clip(priors.screen_rate, 0.03, 0.35))
    if rng.rand() < p_screen:
        air = float(rng.normal(SCREEN_AIR_MEAN, SCREEN_AIR_SD))
    else:
        aDOT = max(2.0, priors.aDOT)
        mu, sg = _lognormal_from_mean_sigma(aDOT, AIR_LN_SIGMA)
        air = float(rng.lognormal(mean=mu, sigma=sg) - 0.8)

    c_base = float(np.clip(priors.comp_exp, 0.50, 0.72))
    p_comp = c_base + float(priors.cpoe) + COMP_B_AIR * (air - priors.aDOT)
    p_comp = float(np.clip(p_comp, 0.35, 0.80))

    a, b = _beta_from_mean_kappa(p_comp, COMP_KAPPA)
    comp = bool(rng.rand() < rng.beta(a, b))
    out["complete"] = comp

    p_int = float(np.clip(INT_RATE_BASE + INT_SLOPE_AIR * max(0.0, air - priors.aDOT), 0.005, 0.06))
    if rng.rand() < p_int:
        out["interception"] = True
        out["turnover"] = True
        out["time_elapsed"] = float(np.clip(rng.normal(6.0, 1.0), 4.0, 12.0))
        return out

    if not comp:
        out["in_bounds"] = False
        out["yards"] = 0.0
        out["time_elapsed"] = float(np.clip(rng.normal(5.5, 1.0), 4.0, 9.0))
        out["air"] = air
        out["yac"] = 0.0
        return out

    yac_mean = max(0.5, priors.yac_mean)
    mu_yac, sg_yac = _lognormal_from_mean_sigma(yac_mean, YAC_LN_SIGMA)
    yac = float(rng.lognormal(mu_yac, sg_yac))

    yards = max(0.0, air + yac)
    out["yards"] = yards
    out["air"] = air
    out["yac"] = yac
    out["in_bounds"] = bool(rng.rand() > 0.10)
    out["time_elapsed"] = float(np.clip(rng.normal(6.5, 1.5), 5.0, 12.0))
    out["fumble"] = bool(rng.rand() < FUMBLE_RATE_PASS)
    out["turnover"] = bool(out["fumble"])
    return out

def sim_run(state: GameState, priors: TeamPriors, rng: np.random.RandomState) -> dict:
    out = {
        "play_type": "run", "yards": 0.0, "stuffed": False,
        "fumble": False, "td": False, "turnover": False,
        "in_bounds": True, "time_elapsed": 0.0, "penalty_yards": 0.0
    }
    # Stuff chance adjusted mildly by heavy box rate
    p_stuff = float(np.clip(STUFF_RATE_BASE + 0.10 * (priors.box_rate_8 - 0.20), 0.08, 0.35))
    if rng.rand() < p_stuff:
        out["stuffed"] = True
        out["yards"] = float(np.clip(rng.normal(-1.0, 1.5), -6.0, 0.0))
    else:
        ypc = max(2.5, priors.rush_ypc)
        mu, sg = _lognormal_from_mean_sigma(ypc, RUN_LN_SIGMA)
        yards = float(rng.lognormal(mu, sg))
        out["yards"] = yards

    out["in_bounds"] = True
    out["time_elapsed"] = float(np.clip(rng.normal(32.0, 4.0), 22.0, 45.0))

    # Fumble (rare)
    out["fumble"] = bool(rng.rand() < FUMBLE_RATE_RUN)
    out["turnover"] = bool(out["fumble"])  # MVP: assume lost
    return out