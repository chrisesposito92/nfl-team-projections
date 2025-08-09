from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .config import (
    REGULATION_QUARTERS, QTR_SECONDS, MAX_PLAYS_HARD, TD_POINTS, FG_POINTS,
    BASE_TEMPO_SEC
)
from .state import GameState, start_state
from .priors import GamePriors, TeamPriors
from .policies import choose_play_type
from .play_models import sim_pass, sim_run
from .special_teams import punt_net_yards, field_goal_attempt, kickoff
from .clock import burn_clock_seconds, next_quarter, yardline_after_gain

# ---- Heuristics for 4th down and special teams (MVP) ----

def _should_fg(state: GameState, priors: TeamPriors) -> bool:
    # FG if inside ~35 yardline (kick <= 52 yards) and 4th & >2
    kick_dist = (100 - state.yardline) + 17
    return (state.down == 4) and (state.distance > 2) and (kick_dist <= 52)

def _should_punt(state: GameState) -> bool:
    # Punt if 4th & long outside FG range and not near midfield
    kick_dist = (100 - state.yardline) + 17
    return (state.down == 4) and (kick_dist > 52) and (state.yardline < 60)

def _should_go_for_it(state: GameState) -> bool:
    # Go for it if 4th & short near midfield or better
    return (state.down == 4) and (state.distance <= 2) and (state.yardline >= 50)

# ---- Core simulation ----

def _record_row(state: GameState, extra: Dict[str, Any]) -> Dict[str, Any]:
    row = state.as_row()
    row.update(extra)
    return row

def _handle_td(state: GameState, plays: list[dict], rng: np.random.RandomState) -> None:
    # Offense just scored
    state.add_points_offense(TD_POINTS)
    # Kickoff from scoring team to the other team
    _do_kickoff(
        state,
        rng,
        plays,
        kicking_team=state.offense,
        receiving_team=state.defense,
    )

def _handle_turnover_on_downs(state: GameState) -> None:
    # Ball goes to defense at current spot
    new_yardline_for_def = 100 - state.yardline
    state.flip_possession(new_yardline=new_yardline_for_def)

def _handle_interception_touchback(state: GameState) -> None:
    # MVP: interceptions become touchbacks; defense starts at own 20/25
    state.flip_possession(new_yardline=20)

def _do_kickoff(
    state: GameState,
    rng: np.random.RandomState,
    plays: list[dict],
    kicking_team: str,
    receiving_team: str,
) -> None:
    """
    Append a kickoff row and set up next possession/spot.
    Handles:
      - Touchbacks (spot 25)
      - Returns (spot drawn around ~27)
      - Rare return TD (awards points to receiving team, triggers next kickoff)
    """
    ko = kickoff(rng)

    # Build a row using current clock/quarter, but override offense/defense to reflect the KICKING team.
    row = state.as_row()
    row.update({
        "offense": kicking_team,
        "defense": receiving_team,
        "play_type": "kickoff",
        "yards_gained": float(ko["yards"]),
        "result": ko["result"],
        "points": 0,
    })
    plays.append(row)

    # Burn a small, outcome-dependent slice of time
    burn_clock_seconds(state, base_seconds=float(ko["time"]), incomplete=False)

    if ko["td"]:
        # Return TD: award points to RECEIVING team (which is currently 'defense' in state)
        # Score fields are offense-perspective; receiving team == state.defense.
        state.score_def += TD_POINTS

        # Kick again from the team that just scored (receiving_team) to the other team.
        # No possession change yet; just execute the next kickoff which WILL set up the next drive.
        _do_kickoff(state, rng, plays, kicking_team=receiving_team, receiving_team=kicking_team)
        return

    # No TD: set receiving team on offense at the spotted yardline.
    start_y = int(ko["start_yardline"])
    # If the receiving team is currently the defense (common case), a single flip sets them on offense.
    if state.defense == receiving_team:
        state.flip_possession(new_yardline=start_y)
    else:
        # Receiving team already on offense (can happen at start of Q3 if offense already equals receiver).
        # Start a new drive with first-and-10 at the new spot.
        state.drive_id += 1
        state.first_down_reset(start_y)

def simulate_single_game(home: str, away: str, year: int, rng: np.random.RandomState,
                         priors: GamePriors | None = None) -> pd.DataFrame:
    if priors is None:
        # Local/lazy import avoids import-time ordering issues
        from .priors import build_game_priors as _build_game_priors
        priors = _build_game_priors(home, away, year)

    st = start_state(home, away, receive_first="away")
    plays: List[Dict[str, Any]] = []

    # Opening kickoff: home kicks to away (since away receives 1H per start_state convention)
    if st.is_kickoff:
        _do_kickoff(st, rng, plays, kicking_team=priors.home.team, receiving_team=priors.away.team)
        st.is_kickoff = False

    while st.quarter <= REGULATION_QUARTERS and len(plays) < MAX_PLAYS_HARD:
        if st.clock <= 0:
            if st.quarter == 2:
                # Halftime -> Q3 kickoff: away kicks to home
                next_quarter(st)  # sets quarter=3, clock=15:00
                _do_kickoff(st, rng, plays, kicking_team=priors.away.team, receiving_team=priors.home.team)
                continue

            if st.quarter >= REGULATION_QUARTERS:
                break

            next_quarter(st)
            continue

        off_pr = priors.away if st.offense == priors.away.team else priors.home

        if _should_fg(st, off_pr):
            good, pts = field_goal_attempt(st.yardline, rng)
            plays.append(_record_row(st, {
                "play_type": "fg", "yards_gained": 0.0,
                "result": "FG GOOD" if good else "FG NO GOOD",
                "points": pts,
                "is_pass": False, "is_run": False,
                "is_complete": False, "is_sack": False,
                "is_interception": False, "is_fumble": False,
            }))
            burn_clock_seconds(st, base_seconds=5.0, incomplete=False)
            if good:
                st.add_points_offense(pts)
                _do_kickoff(st, rng, plays, kicking_team=st.offense, receiving_team=st.defense)
            else:
                st.flip_possession(new_yardline=100 - st.yardline)
            continue

        if _should_punt(st):
            net = punt_net_yards(rng)
            plays.append(_record_row(st, {
                "play_type": "punt", "yards_gained": float(net),
                "result": "PUNT", "points": 0,
                "is_pass": False, "is_run": False,
                "is_complete": False, "is_sack": False,
                "is_interception": False, "is_fumble": False,
            }))
            burn_clock_seconds(st, base_seconds=6.0, incomplete=False)
            st.flip_possession(new_yardline=max(20, int(100 - (st.yardline + net))))
            continue

        # Go-for-it uses normal scrimmage play below
        st.play_id += 1
        ptype = "pass" if choose_play_type(st, off_pr, rng) == "pass" else "run"

        if ptype == "pass":
            outcome = sim_pass(st, off_pr, rng)
            gained = float(outcome["yards"])

            if outcome["interception"]:
                plays.append(_record_row(st, {
                    "play_type": "pass", "yards_gained": gained,
                    "result": "INT", "points": 0,
                    "is_pass": True, "is_run": False,
                    "is_complete": False, "is_sack": False,
                    "is_interception": True, "is_fumble": False,
                }))
                burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=False)
                _handle_interception_touchback(st)
                continue

            if outcome["sack"]:
                new_y = yardline_after_gain(st.yardline, gained)
                plays.append(_record_row(st, {
                    "play_type": "pass", "yards_gained": gained,
                    "result": "SACK", "points": 0,
                    "is_pass": True, "is_run": False,
                    "is_complete": False, "is_sack": True,
                    "is_interception": False, "is_fumble": False,
                }))
                burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=False)
                st.advance_down(new_y, max(0, int(-gained)))
                continue

            if not outcome["complete"]:
                plays.append(_record_row(st, {
                    "play_type": "pass", "yards_gained": 0.0,
                    "result": "INCOMPLETE", "points": 0,
                    "is_pass": True, "is_run": False,
                    "is_complete": False, "is_sack": False,
                    "is_interception": False, "is_fumble": False,
                }))
                burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=True)
                if st.down < 4:
                    st.down += 1
                else:
                    _handle_turnover_on_downs(st)
                continue

            # Completed pass
            new_y = yardline_after_gain(st.yardline, gained)
            if new_y >= 100:
                plays.append(_record_row(st, {
                    "play_type": "pass",
                    "yards_gained": float(100 - st.yardline),
                    "result": "TD", "points": TD_POINTS,
                    "is_pass": True, "is_run": False,
                    "is_complete": True, "is_sack": False,
                    "is_interception": False, "is_fumble": False,
                }))
                burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=False)
                _handle_td(st)
                continue

            if outcome["fumble"]:
                plays.append(_record_row(st, {
                    "play_type": "pass", "yards_gained": gained,
                    "result": "FUMBLE-LOST", "points": 0,
                    "is_pass": True, "is_run": False,
                    "is_complete": True, "is_sack": False,
                    "is_interception": False, "is_fumble": True,
                }))
                burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=False)
                st.flip_possession(new_yardline=100 - new_y)
                continue

            plays.append(_record_row(st, {
                "play_type": "pass", "yards_gained": gained,
                "result": "COMPLETE", "points": 0,
                "is_pass": True, "is_run": False,
                "is_complete": True, "is_sack": False,
                "is_interception": False, "is_fumble": False,
            }))
            burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=False)
            st.advance_down(new_y, int(round(gained)))
            continue

        # Run
        outcome = sim_run(st, off_pr, rng)
        gained = float(outcome["yards"])
        new_y = yardline_after_gain(st.yardline, gained)

        if new_y >= 100:
            plays.append(_record_row(st, {
                "play_type": "run",
                "yards_gained": float(100 - st.yardline),
                "result": "TD", "points": TD_POINTS,
                "is_pass": False, "is_run": True,
                "is_complete": False, "is_sack": False,
                "is_interception": False, "is_fumble": False,
            }))
            burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=False)
            _handle_td(st)
            continue

        if outcome["fumble"]:
            plays.append(_record_row(st, {
                "play_type": "run", "yards_gained": gained,
                "result": "FUMBLE-LOST", "points": 0,
                "is_pass": False, "is_run": True,
                "is_complete": False, "is_sack": False,
                "is_interception": False, "is_fumble": True,
            }))
            burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=False)
            st.flip_possession(new_yardline=100 - new_y)
            continue

        plays.append(_record_row(st, {
            "play_type": "run", "yards_gained": gained,
            "result": "RUN", "points": 0,
            "is_pass": False, "is_run": True,
            "is_complete": False, "is_sack": False,
            "is_interception": False, "is_fumble": False,
        }))
        burn_clock_seconds(st, base_seconds=outcome["time_elapsed"], incomplete=False)
        st.advance_down(new_y, int(round(max(0.0, gained))))

        if st.down > 4:
            _handle_turnover_on_downs(st)

    final = {
        "qtr": st.quarter, "clock": f"{st.clock//60:02d}:{st.clock%60:02d}",
        "offense": st.offense, "defense": st.defense, "down": st.down,
        "dist": st.distance, "yardline": st.yardline, "drive_id": st.drive_id,
        "play_id": st.play_id, "play_type": "END", "result": "END",
        "home": priors.home.team, "away": priors.away.team,
        "home_score": (st.score_def if priors.home.team == st.defense else st.score_off),
        "away_score": (st.score_def if priors.away.team == st.defense else st.score_off),
        "is_pass": False, "is_run": False,
        "is_complete": False, "is_sack": False,
        "is_interception": False, "is_fumble": False,
    }
    df = pd.DataFrame(plays)
    if not df.empty:
        df["home"] = priors.home.team
        df["away"] = priors.away.team
    return pd.concat([df, pd.DataFrame([final])], ignore_index=True)

def simulate_game(
    home: str,
    away: str,
    year: int,
    n_sims: int = 1,
    seed: int = 42,
    priors: Optional[GamePriors] = None
) -> List[pd.DataFrame]:
    rng_base = np.random.RandomState(seed)
    if priors is None:
        priors = build_game_priors(home, away, year)
    sims: List[pd.DataFrame] = []
    for i in range(int(n_sims)):
        rng = np.random.RandomState(rng_base.randint(0, 2**31 - 1))
        sims.append(simulate_single_game(home, away, year, rng=rng, priors=priors))
    return sims