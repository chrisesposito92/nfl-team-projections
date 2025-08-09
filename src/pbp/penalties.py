# src/pbp/penalties.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any


@dataclass
class PenaltyConfig:
    """
    Placeholder penalty config. By default this is inert (enabled=False, base_rate=0.0).
    - base_rate: per-scrimmage-play probability a flag occurs (regardless of type).
    - offense_bias: P(flag on offense) vs defense.
    - yards_choices: generic yardages we might use later (today we only annotate).
    """
    enabled: bool = False
    base_rate: float = 0.0
    offense_bias: float = 0.55
    yards_choices: tuple[int, ...] = (5, 10, 15)


def maybe_penalty(rng: np.random.RandomState, cfg: PenaltyConfig) -> Dict[str, Any]:
    """
    Minimal stub: returns a dict with a penalty annotation. Does NOT change outcomes/yardline.
    This is just a scaffold so downstream can start logging columns & toggling via CLI.
    """
    if not cfg.enabled or cfg.base_rate <= 0.0 or rng.rand() >= cfg.base_rate:
        return {
            "penalty": False,
            "penalty_yards": 0,
            "penalty_on_offense": False,
            "penalty_type": "",
        }

    on_offense = bool(rng.rand() < cfg.offense_bias)
    yards = int(rng.choice(cfg.yards_choices))
    # One generic type for now; we’ll expand with real types/rates later.
    ptype = "GENERIC_OFF" if on_offense else "GENERIC_DEF"

    return {
        "penalty": True,
        "penalty_yards": yards,
        "penalty_on_offense": on_offense,
        "penalty_type": ptype,
    }