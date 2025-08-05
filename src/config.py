"""Configuration settings for the NFL Projections Modeler."""

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Data settings
MIN_TRAINING_YEARS = 3  # Minimum years of historical data needed
POSITIONS = ["QB", "RB", "WR", "TE"]

# Team projection targets
TEAM_TARGETS = [
    "pass_attempts",
    "pass_completions", 
    "pass_yards",
    "pass_tds",
    "rush_attempts",
    "rush_yards",
    "rush_tds"
]

# Player share targets
PLAYER_SHARE_TARGETS = [
    "target_share",
    "rush_attempt_share",
    "pass_td_share",
    "rush_td_share"
]

# Feature engineering settings
ROLLING_WINDOWS = [3, 6, 12]  # Games for rolling averages
MIN_SNAPS = 10  # Minimum snaps to include player in training

# Model parameters
TEAM_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE
}

PLAYER_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE
}

# CLI settings
SEASON_WEEKS = 17  # Regular season weeks