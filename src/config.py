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

SIM_DEFAULT_DRAWS = 1000
SIM_DEFAULT_SEED = 42

SIM_PHI_TARGET = 60.0
SIM_PHI_RUSH = 40.0

SIM_SIGMA_FRAC = {
    "pass_attempts": 0.10,
    "rush_attempts": 0.12,
    "pass_yards": 0.15,
    "rush_yards": 0.18
}

SIM_CALIBRATION_FILE = "artifacts/sim_calibration.json"

SIM_DEFAULT_RHO = [
    [1.0, -0.35, 0.65, -0.25],
    [-0.35, 1.0, -0.25, 0.55],
    [0.65, -0.25, 1.0, -0.20],
    [-0.25, 0.55, -0.20, 1.0],
]

SIM_NB_THETA = {
    "pass_tds": 2.0,
    "rush_tds": 1.5
}

# === Defense & game-script adjustment params ===
ADJUSTMENTS_ENABLED = True
DEF_LOOKBACK_SEASONS = 2  # how many prior seasons to use for defense profile

# Multipliers per 1 std dev of opponent metric
DEF_PASS_YDS_W = 0.06     # pass yards sensitivity to pass EPA allowed
DEF_PASS_ATT_W = 0.03     # pass attempts sensitivity to neutral pass rate allowed
DEF_PASS_TDS_W = 0.08     # pass TD sensitivity to pass EPA allowed

DEF_RUSH_YDS_W = 0.06     # rush yards sensitivity to rush EPA allowed
DEF_RUSH_ATT_W = 0.03     # rush attempts sensitivity to neutral pass rate allowed (negative)
DEF_RUSH_TDS_W = 0.08     # rush TD sensitivity to rush EPA allowed

# Home bonuses (additive to the multiplicative envelope above)
HOME_BONUS_PASS_YDS = 0.02
HOME_BONUS_PASS_ATT = 0.01
HOME_BONUS_RUSH_YDS = 0.01
HOME_BONUS_RUSH_ATT = 0.01

# Numeric stability for rescaling
RESCALE_EPS = 1e-9