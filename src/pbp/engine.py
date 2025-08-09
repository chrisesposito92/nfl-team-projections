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
    priors: Optional[GamePriors] = None,
) -> List[pd.DataFrame]:
    """
    Lightweight per-play sim used by the CLI.
    Phase-1 additions:
      - is_penalty / penalty_yards placeholders (no types yet)
      - Option B++ 'soft' anchors (plays + rush rate) if available
      - Incomplete clock logic already wired; OOB flag still not affecting time
    """
    rng = np.random.RandomState(seed)
    sims: List[pd.DataFrame] = []

    # Team priors
    if priors is None:
        priors = build_game_priors(home, away, year)

    # Try to get Option B++ anchors; fall back if missing
    anchors = maybe_build_game_anchors(home, away, year)

    # Pace and run-rate baselines
    if anchors is not None:
        pace_plays = int(np.clip(anchors["home"]["plays"] + anchors["away"]["plays"], 90, 160))
        run_rate_home = float(np.clip(anchors["home"]["rush_rate"], 0.20, 0.70))
        run_rate_away = float(np.clip(anchors["away"]["rush_rate"], 0.20, 0.70))
    else:
        pace_plays = 120
        run_rate_home = 0.42
        run_rate_away = 0.40

    # Pass mechanics from priors
    p_sack_base_home = float(np.clip(priors.home.sack_rate, 0.03, 0.12))
    p_sack_base_away = float(np.clip(priors.away.sack_rate, 0.03, 0.12))
    p_comp_home = float(np.clip(priors.home.comp_exp + priors.home.cpoe, 0.50, 0.75))
    p_comp_away = float(np.clip(priors.away.comp_exp + priors.away.cpoe, 0.50, 0.75))

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
            # end-of-game guard
            if qtr > 4:
                break

            is_home_off = (off == home)
            run_rate = run_rate_home if is_home_off else run_rate_away
            p_sack_base = p_sack_base_home if is_home_off else p_sack_base_away
            p_comp = p_comp_home if is_home_off else p_comp_away

            long = (dist >= 8)
            # slightly adaptive run rate
            adj_run_rate = np.clip(run_rate - 0.10 * long + 0.05 * (down == 1), 0.15, 0.80)
            call_run = (rng.rand() < adj_run_rate)

            play_type = "run"
            is_sack = False
            is_complete = False
            is_interception = False
            is_fumble = False
            yards = 0.0
            result = "RUN"

            if not call_run:
                play_type = "pass"
                # sack pressure a bit higher on long
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
                        # rare fumble after catch (placeholder rate)
                        is_fumble = bool(rng.rand() < 0.01)
                        if is_fumble:
                            result = "FUMBLE-LOST"
                    else:
                        yards = 0.0
                        result = "INCOMPLETE"
                        # rare int on incompletion (placeholder)
                        is_interception = bool(rng.rand() < 0.02)
                        if is_interception:
                            result = "INT"
            else:
                mu = 4.3 - 1.0 * long + 0.7 * (down == 1)
                sd = 3.0
                yards = float(np.clip(rng.normal(mu, sd), -4.0, 35.0))
                result = "RUN"
                # rare fumble on run
                is_fumble = bool(rng.rand() < 0.01)
                if is_fumble:
                    result = "FUMBLE-LOST"

            # ----- Penalty (placeholder) -----
            is_penalty = False
            penalty_on = ""
            penalty_yards = 0.0
            auto_first = False
            if rng.rand() < PENALTY_RATE:
                is_penalty = True
                if rng.rand() < 0.5:
                    # defensive penalty
                    penalty_on = "defense"
                    penalty_yards = float(PENALTY_DEFENSE_YDS)
                    auto_first = bool(rng.rand() < PENALTY_DEF_AUTO_FIRST_PROB and play_type == "pass")
                else:
                    # offensive penalty
                    penalty_on = "offense"
                    penalty_yards = float(PENALTY_OFFENSE_YDS)

            # yardline movement (play yards + penalty yards; simplified enforcement)
            nonneg_gain = max(0.0, yards)
            total_gain = nonneg_gain + (penalty_yards if is_penalty else 0.0)
            new_yardline = float(np.clip(yardline + total_gain, 1.0, 99.0))

            # first down logic (simple; auto-first on some defensive penalties)
            got_first_by_play = (nonneg_gain >= dist)
            got_first_by_pen = bool(is_penalty and penalty_on == "defense" and (auto_first or penalty_yards >= dist))
            first_down = got_first_by_play or got_first_by_pen

            points = 0.0
            # TD on the play (penalties can still bring you short in real life; placeholder allows it)
            if new_yardline >= 100.0:
                points = 7.0
                new_yardline = 25.0
                down = 1
                dist = 10.0
                drive_id += 1
                off, defn = defn, off
            else:
                # basic clock burn
                # incomplete passes already zero yards; we still burn generic time here.
                # (OOB not affecting clock yet per scope.)
                clock -= int(np.clip(np.random.normal(25, 8), 15, 45))
                if is_penalty:
                    clock -= int(PENALTY_TIME_SEC)

                if clock <= 0:
                    qtr += 1
                    clock = 15 * 60
                    if qtr > 4:
                        # end row
                        rows.append({
                            "qtr": 4, "clock": "00:00", "offense": off, "defense": defn,
                            "down": int(down), "dist": float(dist), "yardline": int(yardline),
                            "drive_id": int(drive_id), "play_id": int(play_id),
                            "play_type": "END", "yards_gained": np.nan, "result": "END",
                            "points": 0.0, "home": home, "away": away,
                            "home_score": home_score, "away_score": away_score,
                            "is_pass": False, "is_run": False,
                            "is_complete": False, "is_sack": False,
                            "is_interception": False, "is_fumble": False,
                            "is_penalty": False, "penalty_on": "", "penalty_yards": 0.0,
                        })
                        break

                # turnover on downs (placeholder) if 4th & fail
                if (play_type in ("run", "pass")) and not first_down and down == 4:
                    # change possession at current spot (post-penalty)
                    off, defn = defn, off
                    new_yardline = 100.0 - new_yardline
                    down = 1
                    dist = 10.0
                    drive_id += 1
                else:
                    # advance or set next down/dist
                    if first_down:
                        down = 1
                        dist = 10.0
                    else:
                        down = 4 if down == 3 else (down + 1)
                        dist = max(1.0, dist - nonneg_gain)

            # scoring
            if off == home:
                home_score += points
            else:
                away_score += points

            rows.append({
                "qtr": int(qtr),
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
                "is_pass": (play_type == "pass"),
                "is_run": (play_type == "run"),
                "is_complete": bool(is_complete) if play_type == "pass" else False,
                "is_sack": bool(is_sack) if play_type == "pass" else False,
                "is_interception": bool(is_interception),
                "is_fumble": bool(is_fumble),
                "is_penalty": bool(is_penalty),
                "penalty_on": penalty_on if is_penalty else "",
                "penalty_yards": float(penalty_yards) if is_penalty else 0.0,
            })

            yardline = int(max(1, min(99, new_yardline)))
            play_id += 1

        sims.append(pd.DataFrame(rows))

    return sims