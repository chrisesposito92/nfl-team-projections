from __future__ import annotations

# --------- General ----------
DEFAULT_SEED = 2025
DEFAULT_SIMS = 100

REGULATION_QUARTERS = 4
QTR_SECONDS = 15 * 60
MAX_PLAYS_HARD = 230  # guardrail
KICK_TOUCHBACK_YD = 25

# --------- Tempo / clock (seconds between snaps) ----------
BASE_TEMPO_SEC = 28.0
HURRY_TEMPO_SEC = 18.0
NOHUDDLE_TEMPO_SEC = 24.0

# Incomplete pass costs ~6 seconds (play time) but otherwise does not burn play clock.
INCOMPLETE_PLAY_TIME = 6.0

# --------- Play-call policy (pass vs run) ----------
BASE_PASS_RATE = 0.57
PASS_RATE_SLOPE_LONG_YTG = 0.23   # +pass when distance is long
PASS_RATE_SLOPE_POS_FIELD = 0.08  # +pass when backed up (own side)
PASS_RATE_SLOPE_TRAIL = 0.18      # +pass when trailing (score pressure)
PASS_RATE_SLOPE_CLOCK = 0.22      # +pass late (time pressure)

LONG_YTG_THRESH = 7
FIELD_POS_CENTER = 50

# --------- Passing model ----------
SACK_RATE_BASE = 0.07
SACK_YARDS_MEAN = -6.5
SACK_YARDS_SD = 3.0

AIR_LN_SIGMA = 0.55         # volatility of air yards (non-screen)
SCREEN_PROB_BASE = 0.12     # probability the pass is a screen
SCREEN_AIR_MEAN = -1.2      # typical screen thrown behind LOS
SCREEN_AIR_SD = 1.0

# Completion probability: start at E[comp%], add CPOE, then adjust by air yards deviation
COMP_KAPPA = 40.0           # strength for Beta(alpha, beta) around the mean completion prob
COMP_B_AIR = -0.06          # completion slope vs (air - aDOT); negative => deeper throws are harder

# Interception rate increases with depth beyond aDOT
INT_RATE_BASE = 0.025
INT_SLOPE_AIR = 0.002

# YAC lognormal
YAC_LN_SIGMA = 0.45

# --------- Rushing model ----------
STUFF_RATE_BASE = 0.18
RUN_LN_SIGMA = 0.40
FUMBLE_RATE_RUN = 0.010
FUMBLE_RATE_PASS = 0.015

# --------- Special teams ----------
PUNT_NET_YDS_MEAN = 42.0
PUNT_NET_YDS_SD = 5.5

# FG: logistic on distance (yardline to middle of endzone + 17)
FG_B0 = 7.0     # tuned so ~95% at 20 yds, ~65% at 50 yds (coarse MVP)
FG_B_DIST = -0.13

# PAT handling (MVP): award 7 for TD (skip XP/2pt modeling)
TD_POINTS = 7
FG_POINTS = 3
SAFETY_POINTS = 2  # not modeled in MVP, placeholder

# --------- Penalties (Phase-1 placeholder) ----------
# Simple, global rates. We don't model types yet; just side + yards.
PENALTY_RATE = 0.06            # per scrimmage play
PENALTY_OFFENSE_YDS = -10      # assessed from previous spot (sign from offense POV)
PENALTY_DEFENSE_YDS = 5        # generic defensive penalty (no DPI special case yet)
PENALTY_DEF_AUTO_FIRST_PROB = 0.10  # chance a defensive penalty grants automatic first down
PENALTY_TIME_SEC = 5.0         # extra seconds to burn when a penalty is enforced