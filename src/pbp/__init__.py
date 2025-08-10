# src/pbp/__init__.py

from .engine import simulate_game, simulate_single_game
from .priors import build_game_priors, TeamPriors, GamePriors, get_team_priors

# Lazy wrappers so importing this package doesn't drag heavy dependencies
def attribute_players_from_plays(*args, **kwargs):
    from .players import attribute_players_from_plays as _fn
    return _fn(*args, **kwargs)

def build_game_sampling(*args, **kwargs):
    from .players import build_game_sampling as _fn
    return _fn(*args, **kwargs)

__all__ = [
    "simulate_game",
    "simulate_single_game",
    "build_game_priors",
    "get_team_priors",
    "TeamPriors",
    "GamePriors",
    "attribute_players_from_plays",
    "build_game_sampling",
]

from .calibrate import run_calibration, simulate_games_for_schedule, summarize_quantiles, reliability_report, pit_report

__all__ += [
    "run_calibration",
    "simulate_games_for_schedule",
    "summarize_quantiles",
    "reliability_report",
    "pit_report",
]