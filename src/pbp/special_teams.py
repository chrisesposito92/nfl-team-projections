from __future__ import annotations
import numpy as np

from .config import PUNT_NET_YDS_MEAN, PUNT_NET_YDS_SD, FG_B0, FG_B_DIST, FG_POINTS

# Modern-ish touchback rates are high; you can tune these per-season later.
_KO_TB_PROB = 0.62           # P(touchback)
_KO_RET_START_MEAN = 27.0     # mean starting yardline on returns (from receiving goal line)
_KO_RET_START_SD = 7.0        # sd of starting yardline on returns
_KO_RET_START_MIN = 10.0      # floor for return start
_KO_RET_START_MAX = 50.0      # ceiling (rare long returns outside TD)
_KO_RET_TD_PROB = 0.0015      # very rare KR TD


def _clip(x: float, lo: float, hi: float) -> float:
    import numpy as np
    return float(np.clip(x, lo, hi))

def kickoff(rng) -> dict:
    """
    Simulate a kickoff.
    Returns dict with:
      - 'touchback': bool
      - 'td': bool
      - 'start_yardline': int (receiving team's yardline 1..99; 25 on TB)
      - 'yards': float (for logging only; ~ start_yardline - 25 on returns; 0 on TB)
      - 'time': float seconds to burn on clock
      - 'result': str in {"TB","RETURN","RET-TD"}
    """
    import numpy as np

    # Touchback
    if rng.rand() < _KO_TB_PROB:
        return {
            "touchback": True,
            "td": False,
            "start_yardline": 25,
            "yards": 0.0,
            "time": _clip(rng.normal(6.0, 1.0), 4.0, 9.0),
            "result": "TB",
        }

    # Very rare immediate TD on kick return
    if rng.rand() < _KO_RET_TD_PROB:
        # TD, no starting spot (we'll kickoff again from the return team)
        return {
            "touchback": False,
            "td": True,
            "start_yardline": 100,  # sentinel; not used for spotting
            "yards": 100.0,
            "time": _clip(rng.normal(12.0, 2.0), 9.0, 16.0),
            "result": "RET-TD",
        }

    # Normal return: draw final starting yardline directly (stable, easy to calibrate)
    start = int(round(_clip(rng.normal(_KO_RET_START_MEAN, _KO_RET_START_SD),
                            _KO_RET_START_MIN, _KO_RET_START_MAX)))
    return {
        "touchback": False,
        "td": False,
        "start_yardline": max(1, min(99, start)),
        "yards": float(start - 25),  # relative to touchback baseline for logging
        "time": _clip(rng.normal(10.0, 2.0), 6.0, 16.0),
        "result": "RETURN",
    }

def punt_net_yards(rng: np.random.RandomState) -> int:
    return int(np.clip(rng.normal(PUNT_NET_YDS_MEAN, PUNT_NET_YDS_SD), 28.0, 65.0))

def fg_make_prob(distance_yards: int) -> float:
    # Logistic: p = 1 / (1 + exp(-(b0 + b1 * dist)))
    z = FG_B0 + FG_B_DIST * float(distance_yards)
    return float(1.0 / (1.0 + np.exp(-z)))

def field_goal_attempt(spot_yardline: int, rng: np.random.RandomState) -> tuple[bool, int]:
    """
    spot_yardline: offense yardline (1..99), goal at 100; kick distance ~= (100 - yardline) + 17
    Returns (good, points)
    """
    dist = (100 - int(spot_yardline)) + 17
    p = fg_make_prob(dist)
    good = bool(rng.rand() < p)
    return good, FG_POINTS if good else 0