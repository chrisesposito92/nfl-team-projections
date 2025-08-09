from __future__ import annotations
import numpy as np

from .config import (
    BASE_PASS_RATE, PASS_RATE_SLOPE_LONG_YTG, PASS_RATE_SLOPE_POS_FIELD,
    PASS_RATE_SLOPE_TRAIL, PASS_RATE_SLOPE_CLOCK, LONG_YTG_THRESH, FIELD_POS_CENTER
)
from .state import GameState
from .priors import TeamPriors

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def pass_prob(state: GameState, priors: TeamPriors) -> float:
    """
    Simple contextual pass probability:
      - baseline team pass rate
      - + when distance is long
      - + when trailing
      - + when late in half/game
      - + when backed up (own half)
    """
    base = float(priors.pass_rate)
    logit = np.log(base / max(1e-6, 1 - base))

    # Long distance
    if state.distance >= LONG_YTG_THRESH:
        logit += PASS_RATE_SLOPE_LONG_YTG

    # Field position (more conservative in plus territory, slightly more pass when backed up)
    field_bias = -PASS_RATE_SLOPE_POS_FIELD * ((state.yardline - FIELD_POS_CENTER) / 50.0)
    logit += field_bias

    # Score pressure (trailing => more pass)
    score_diff = state.score_off - state.score_def
    if score_diff < 0:
        logit += PASS_RATE_SLOPE_TRAIL * min(1.0, abs(score_diff) / 10.0)

    # Time pressure (late quarters)
    sec_left_game = (4 - state.quarter) * 900 + state.clock
    if sec_left_game < 6 * 60:
        logit += PASS_RATE_SLOPE_CLOCK * (1.0 - sec_left_game / (6 * 60))

    return float(_sigmoid(logit))

def choose_play_type(state: GameState, priors: TeamPriors, rng: np.random.RandomState) -> str:
    p_pass = pass_prob(state, priors)
    return "pass" if rng.rand() < p_pass else "run"