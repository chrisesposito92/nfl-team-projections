from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .config import (
    REGULATION_QUARTERS, QTR_SECONDS, MAX_PLAYS_HARD, TD_POINTS, FG_POINTS,
    BASE_TEMPO_SEC
)
from .config import PENALTY_RATE, PENALTY_OFFENSE_YDS, PENALTY_DEFENSE_YDS, PENALTY_DEF_AUTO_FIRST_PROB, PENALTY_TIME_SEC
from .state import GameState, start_state
from .priors import GamePriors, TeamPriors, build_game_priors, maybe_build_game_anchors
from .policies import choose_play_type
from .play_models import sim_pass, sim_run
from .special_teams import punt_net_yards, field_goal_attempt, kickoff
from .clock import burn_clock_seconds, next_quarter, yardline_after_gain
from .penalties import PenaltyConfig, maybe_penalty
from .anchor import soft_anchor_yards, AnchorWeights

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

def simulate_single_game(
    home: str,
    away: str,
    year: int,
    rng: np.random.RandomState,
    priors: GamePriors | None = None,
    # NEW knobs to mirror simple engine
    penalties_enabled: bool = False,
    penalty_rate: float = 0.0,
    apply_anchor: bool = False,
    anchor_weight: float = 0.0,
) -> pd.DataFrame:
    if priors is None:
        from .priors import build_game_priors as _build_game_priors
        priors = _build_game_priors(home, away, year)

    # NEW: penalty config
    pen_cfg = PenaltyConfig(enabled=bool(penalties_enabled),
                            base_rate=float(max(0.0, penalty_rate)))

    st = start_state(home, away, receive_first="away")
    plays: List[Dict[str, Any]] = []

    # Opening kickoff
    if st.is_kickoff:
        _do_kickoff(st, rng, plays, kicking_team=priors.home.team, receiving_team=priors.away.team)
        st.is_kickoff = False

    while st.quarter <= REGULATION_QUARTERS and len(plays) < MAX_PLAYS_HARD:
        if st.clock <= 0:
            if st.quarter == 2:
                next_quarter(st)
                _do_kickoff(st, rng, plays, kicking_team=priors.away.team, receiving_team=priors.home.team)
                continue
            if st.quarter >= REGULATION_QUARTERS:
                break
            next_quarter(st)
            continue

        off_pr = priors.away if st.offense == priors.away.team else priors.home

        if _should_fg(st, off_pr):
            good, pts = field_goal_attempt(st.yardline, rng)
            row = _record_row(st, {
                "play_type": "fg", "yards_gained": 0.0,
                "result": "FG GOOD" if good else "FG NO GOOD",
                "points": pts,
                "is_pass": False, "is_run": False,
                "is_complete": False, "is_sack": False,
                "is_interception": False, "is_fumble": False,
                # penalty stub columns (special teams rarely flagged here in MVP)
                "penalty": False, "penalty_yards": 0,
                "penalty_on_offense": False, "penalty_type": "",
            })
            plays.append(row)
            burn_clock_seconds(st, base_seconds=5.0, incomplete=False)
            if good:
                st.add_points_offense(pts)
                _do_kickoff(st, rng, plays, kicking_team=st.offense, receiving_team=st.defense)
            else:
                st.flip_possession(new_yardline=100 - st.yardline)
            continue

        if _should_punt(st):
            net = punt_net_yards(rng)
            row = _record_row(st, {
                "play_type": "punt", "yards_gained": float(net),
                "result": "PUNT", "points": 0,
                "is_pass": False, "is_run": False,
                "is_complete": False, "is_sack": False,
                "is_interception": False, "is_fumble": False,
                "penalty": False, "penalty_yards": 0,
                "penalty_on_offense": False, "penalty_type": "",
            })
            plays.append(row)
            burn_clock_seconds(st, base_seconds=6.0, incomplete=False)
            st.flip_possession(new_yardline=max(20, int(100 - (st.yardline + net))))
            continue

        # Go-for-it uses scrimmage play below
        st.play_id += 1
        ptype = "pass" if choose_play_type(st, off_pr, rng) == "pass" else "run"

        if ptype == "pass":
            outcome = sim_pass(st, off_pr, rng)
            gained = float(outcome["yards"])

            if outcome["interception"]:
                pen = maybe_penalty(rng, pen_cfg)
                plays.append(_record_row(st, {
                    "play_type": "pass", "yards_gained": gained,
                    "result": "INT", "points": 0,
                    "is_pass": True, "is_run": False,
                    "is_complete": False, "is_sack": False,
                    "is_interception": True, "is_fumble": False,
                    "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
                    "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
                }))
                burn_clock_seconds(st, base_seconds=BASE_TEMPO_SEC, incomplete=False)
                _handle_interception_touchback(st)
                continue

            if outcome["sack"]:
                new_y = yardline_after_gain(st.yardline, gained)
                pen = maybe_penalty(rng, pen_cfg)
                plays.append(_record_row(st, {
                    "play_type": "pass", "yards_gained": gained,
                    "result": "SACK", "points": 0,
                    "is_pass": True, "is_run": False,
                    "is_complete": False, "is_sack": True,
                    "is_interception": False, "is_fumble": False,
                    "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
                    "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
                }))
                burn_clock_seconds(st, base_seconds=BASE_TEMPO_SEC, incomplete=False)
                st.advance_down(new_y, max(0, int(-gained)))
                continue

            if not outcome["complete"]:
                pen = maybe_penalty(rng, pen_cfg)
                plays.append(_record_row(st, {
                    "play_type": "pass", "yards_gained": 0.0,
                    "result": "INCOMPLETE", "points": 0,
                    "is_pass": True, "is_run": False,
                    "is_complete": False, "is_sack": False,
                    "is_interception": False, "is_fumble": False,
                    "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
                    "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
                }))
                burn_clock_seconds(st, base_seconds=BASE_TEMPO_SEC, incomplete=True)
                if st.down < 4:
                    st.down += 1
                else:
                    _handle_turnover_on_downs(st)
                continue

            # Completed pass
            new_y = yardline_after_gain(st.yardline, gained)
            if new_y >= 100:
                pen = maybe_penalty(rng, pen_cfg)
                plays.append(_record_row(st, {
                    "play_type": "pass",
                    "yards_gained": float(100 - st.yardline),
                    "result": "TD", "points": TD_POINTS,
                    "is_pass": True, "is_run": False,
                    "is_complete": True, "is_sack": False,
                    "is_interception": False, "is_fumble": False,
                    "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
                    "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
                }))
                burn_clock_seconds(st, base_seconds=BASE_TEMPO_SEC, incomplete=False)
                _handle_td(st, plays, rng)
                continue

            if outcome["fumble"]:
                pen = maybe_penalty(rng, pen_cfg)
                plays.append(_record_row(st, {
                    "play_type": "pass", "yards_gained": gained,
                    "result": "FUMBLE-LOST", "points": 0,
                    "is_pass": True, "is_run": False,
                    "is_complete": True, "is_sack": False,
                    "is_interception": False, "is_fumble": True,
                    "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
                    "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
                }))
                burn_clock_seconds(st, base_seconds=BASE_TEMPO_SEC, incomplete=False)
                st.flip_possession(new_yardline=100 - new_y)
                continue

            pen = maybe_penalty(rng, pen_cfg)
            plays.append(_record_row(st, {
                "play_type": "pass", "yards_gained": gained,
                "result": "COMPLETE", "points": 0,
                "is_pass": True, "is_run": False,
                "is_complete": True, "is_sack": False,
                "is_interception": False, "is_fumble": False,
                "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
                "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
            }))
            burn_clock_seconds(
                st,
                base_seconds=BASE_TEMPO_SEC,
                incomplete=False,
                oob=(not bool(outcome.get("in_bounds", True))),
            )
            st.advance_down(new_y, int(round(gained)))
            continue

        # Run
        outcome = sim_run(st, off_pr, rng)
        gained = float(outcome["yards"])
        new_y = yardline_after_gain(st.yardline, gained)

        if new_y >= 100:
            pen = maybe_penalty(rng, pen_cfg)
            plays.append(_record_row(st, {
                "play_type": "run",
                "yards_gained": float(100 - st.yardline),
                "result": "TD", "points": TD_POINTS,
                "is_pass": False, "is_run": True,
                "is_complete": False, "is_sack": False,
                "is_interception": False, "is_fumble": False,
                "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
                "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
            }))
            burn_clock_seconds(st, base_seconds=BASE_TEMPO_SEC, incomplete=False)
            _handle_td(st, plays, rng)
            continue

        if outcome["fumble"]:
            pen = maybe_penalty(rng, pen_cfg)
            plays.append(_record_row(st, {
                "play_type": "run", "yards_gained": gained,
                "result": "FUMBLE-LOST", "points": 0,
                "is_pass": False, "is_run": True,
                "is_complete": False, "is_sack": False,
                "is_interception": False, "is_fumble": True,
                "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
                "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
            }))
            burn_clock_seconds(st, base_seconds=BASE_TEMPO_SEC, incomplete=False)
            st.flip_possession(new_yardline=100 - new_y)
            continue

        pen = maybe_penalty(rng, pen_cfg)
        plays.append(_record_row(st, {
            "play_type": "run", "yards_gained": gained,
            "result": "RUN", "points": 0,
            "is_pass": False, "is_run": True,
            "is_complete": False, "is_sack": False,
            "is_interception": False, "is_fumble": False,
            "penalty": bool(pen["penalty"]), "penalty_yards": int(pen["penalty_yards"]),
            "penalty_on_offense": bool(pen["penalty_on_offense"]), "penalty_type": str(pen["penalty_type"]),
        }))
        burn_clock_seconds(st, base_seconds=BASE_TEMPO_SEC, incomplete=False)
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
        # schema consistency
        "penalty": False, "penalty_yards": 0,
        "penalty_on_offense": False, "penalty_type": "",
    }
    df = pd.DataFrame(plays)
    if not df.empty:
        df["home"] = priors.home.team
        df["away"] = priors.away.team

    # NEW: optional soft anchor (post-process, yards only)
    if apply_anchor and anchor_weight > 0.0 and not df.empty:
        df = soft_anchor_yards(df, priors=priors, weights=AnchorWeights(yards_weight=float(anchor_weight)))

    return pd.concat([df, pd.DataFrame([final])], ignore_index=True)

def simulate_game(
    home: str,
    away: str,
    year: int,
    n_sims: int = 1,
    seed: int = 42,
    priors: GamePriors | None = None,
    # NEW knobs (Phase-1 stubs, defaults are inert)
    penalties_enabled: bool = False,
    penalty_rate: float = 0.0,
    anchor_weight: float = 0.0,
) -> list[pd.DataFrame]:
    """
    Returns a list of play-by-play DataFrames (length n_sims).
    Adds penalty stub columns and optional soft-anchoring of pass/rush yards.
    """
    rng = np.random.RandomState(seed)
    sims: list[pd.DataFrame] = []

    # Fall back to default priors if caller didn't pass any
    if priors is None:
        from .priors import build_game_priors
        priors = build_game_priors(home, away, year)

    # Simple knobs we used earlier; keep existing behaviour
    pace_plays = 120
    run_rate_home = 0.42
    run_rate_away = 0.40
    p_sack_base_home = 0.065
    p_sack_base_away = 0.065
    p_comp_home = 0.635
    p_comp_away = 0.625

    pen_cfg = PenaltyConfig(
        enabled=bool(penalties_enabled),
        base_rate=float(max(0.0, penalty_rate)),
    )

    for _ in range(n_sims):
        rows = []
        off = away
        defn = home
        home_score = 0.0
        away_score = 0.0
        clock = 15 * 60
        qtr = 1
        yardline = 25
        down = 1
        dist = 10
        drive_id = 1
        play_id = 1

        for __ in range(int(pace_plays)):
            is_home_off = (off == home)
            run_rate = run_rate_home if is_home_off else run_rate_away
            p_sack_base = p_sack_base_home if is_home_off else p_sack_base_away
            p_comp = p_comp_home if is_home_off else p_comp_away

            long = (dist >= 8)
            adj_run_rate = np.clip(run_rate - 0.10 * long + 0.05 * (down == 1), 0.15, 0.80)
            call_run = (rng.rand() < adj_run_rate)

            play_type = "run"
            is_pass = False
            is_run = False
            is_sack = False
            is_complete = False
            is_interception = False
            is_fumble = False
            yards = 0.0
            result = "RUN"

            # Penalty annotation (stub; does not modify outcome)
            pen = maybe_penalty(rng, pen_cfg)

            if not call_run:
                play_type = "pass"
                is_pass = True
                p_sack = np.clip(p_sack_base * (1.00 + 0.25 * long), 0.01, 0.20)
                if rng.rand() < p_sack:
                    is_sack = True
                    yards = -float(np.clip(rng.normal(6.5, 2.0), 2.0, 12.0))
                    result = "SACK"
                else:
                    is_complete = (rng.rand() < p_comp)
                    if is_complete:
                        mu = 6.5 + 6.0 * long
                        sd = 6.0 + 3.0 * long
                        yards = float(np.clip(rng.normal(mu, sd), -3.0, 55.0))
                        result = "COMPLETE"
                    else:
                        yards = 0.0
                        result = "INCOMPLETE"
            else:
                is_run = True
                mu = 4.3 - 1.0 * long + 0.7 * (down == 1)
                sd = 3.0
                yards = float(np.clip(rng.normal(mu, sd), -4.0, 35.0))
                result = "RUN"

            gained = yards
            new_yardline = yardline + max(0.0, gained)
            first_down = (gained >= dist)
            if first_down:
                down = 1
                dist = 10
            else:
                down = 4 if down == 3 else (down + 1)
                dist = max(1.0, dist - max(0.0, gained))

            points = 0.0
            if new_yardline >= 100:
                points = 7.0
                new_yardline = 25
                down = 1
                dist = 10
                drive_id += 1
                off, defn = defn, off
            else:
                # Keep clock behaviour consistent with your current MVP
                clock -= int(np.clip(np.random.normal(25, 8), 15, 45))
                if clock <= 0:
                    qtr += 1
                    clock = 15 * 60
                    if qtr > 4:
                        rows.append({
                            "qtr": 4, "clock": "00:00", "offense": off, "defense": defn,
                            "down": down, "dist": dist, "yardline": int(yardline),
                            "drive_id": drive_id, "play_id": play_id, "play_type": "END",
                            "yards_gained": np.nan, "result": "END", "points": 0.0,
                            "home": home, "away": away, "home_score": home_score, "away_score": away_score,
                            "is_pass": False, "is_run": False, "is_complete": False, "is_sack": False,
                            "is_interception": False, "is_fumble": False,
                            # penalty stub columns:
                            "penalty": False, "penalty_yards": 0, "penalty_on_offense": False, "penalty_type": "",
                        })
                        break

                if (play_type in ("run", "pass")) and down == 4 and not first_down:
                    off, defn = defn, off
                    new_yardline = 100 - new_yardline
                    down = 1
                    dist = 10
                    drive_id += 1

            if off == home:
                home_score += points
            else:
                away_score += points

            rows.append({
                "qtr": qtr,
                "clock": f"{int(clock // 60):02d}:{int(clock % 60):02d}",
                "offense": off,
                "defense": defn,
                "down": int(down),
                "dist": float(dist),
                "yardline": int(max(1, min(99, new_yardline))),
                "drive_id": int(drive_id),
                "play_id": int(play_id),
                "play_type": play_type,
                "yards_gained": float(yards),
                "result": result,
                "points": float(points),
                "home": home,
                "away": away,
                "home_score": float(home_score),
                "away_score": float(away_score),
                "is_pass": bool(is_pass),
                "is_run": bool(is_run),
                "is_complete": bool(is_complete),
                "is_sack": bool(is_sack),
                "is_interception": bool(is_interception),
                "is_fumble": bool(is_fumble),
                # penalty stub annotations:
                "penalty": bool(pen["penalty"]),
                "penalty_yards": int(pen["penalty_yards"]),
                "penalty_on_offense": bool(pen["penalty_on_offense"]),
                "penalty_type": str(pen["penalty_type"]),
            })
            yardline = int(max(1, min(99, new_yardline)))
            play_id += 1

        df = pd.DataFrame(rows)

        # Optional soft-anchoring (post-process, yards only)
        if anchor_weight and anchor_weight > 0.0 and not df.empty:
            df = soft_anchor_yards(df, priors=priors, weights=AnchorWeights(yards_weight=float(anchor_weight)))

        sims.append(df)

    return sims