SEED = 42
START_SEASON = 2011
ROLLING_GAMES = 8
SHARE_ROLLING_GAMES = 4

TEAM_TARGETS = [
    "pass_attempts",
    "pass_yards",
    "pass_tds",
    "rush_attempts",
    "rush_yards",
    "rush_tds",
]

RECEIVING_POS = ["WR", "TE", "RB"]
RUSHING_POS = ["RB", "QB", "WR", "TE"]
OFFENSE_POS = ["QB", "RB", "WR", "TE"]
DEPTH_CHART_LIMITS = {"QB": 1, "RB": 3, "WR": 4, "TE": 2}

ARTIFACTS_DIR = "artifacts"
TEAM_MODEL_DIRNAME = "team_models"
SHARE_MODEL_DIRNAME = "share_models"