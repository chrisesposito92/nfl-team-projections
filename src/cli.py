from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd

from .config import (
    TEAM_TARGETS,
    SIM_DEFAULT_DRAWS,
    SIM_DEFAULT_SEED,
    ARTIFACTS_DIR,
    PHI_PASS_MIN,
    PHI_PASS_MAX,
    PHI_RUSH_MIN,
    PHI_RUSH_MAX,
)
from .data import import_team_list, team_schedule_for_year
from .predict import project_team_week, project_team_season
from .utils import set_display
from .simulate import simulate_from_point, _sigma_profile_cached
from .adjust import _rz_profile_cached


def _get_phi_sigma(team: str, year: int) -> tuple[float, float, float]:
    try:
        prof = _rz_profile_cached(year)
        if team in prof.index:
            phi_pass = float(prof.loc[team, "phi_pass_td_rz"])
            phi_rush = float(prof.loc[team, "phi_rush_td_rz"])
        else:
            phi_pass = 1.0
            phi_rush = 1.0
    except Exception:
        phi_pass = 1.0
        phi_rush = 1.0
    phi_pass = float(np.clip(phi_pass, PHI_PASS_MIN, PHI_PASS_MAX))
    phi_rush = float(np.clip(phi_rush, PHI_RUSH_MIN, PHI_RUSH_MAX))

    try:
        sprof = _sigma_profile_cached(year)
        sigma_td = float(sprof.loc[team, "sigma_td"]) if team in sprof.index else 1.0
    except Exception:
        sigma_td = 1.0

    return phi_pass, phi_rush, sigma_td


def main():
    set_display()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("NFL Offensive Projections Modeler (V1.0)")

    try:
        year = int(input("Enter projection year (e.g., 2025): ").strip())
    except Exception:
        print("Invalid year.")
        sys.exit(1)

    teams = import_team_list()
    print("Valid teams:", ", ".join(teams))
    team = input("Select a team abbreviation (e.g., CIN): ").strip().upper()
    if team not in teams:
        print("Invalid team.")
        sys.exit(1)

    full = input("Project full REGULAR season? (yes/no): ").strip().lower()
    if full in ("y", "yes"):
        print("Generating full-season projections...")
        phi_p, phi_r, sigma_td = _get_phi_sigma(team, year)
        print(f"Adjustments -> RZ φ_pass={phi_p:.3f}, φ_rush={phi_r:.3f}; TD σ={sigma_td:.3f}")

        proj = project_team_season(year, team)
        cols = [
            "season", "team", "week", "player_name", "position",
            "proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds",
            "proj_rush_att", "proj_rush_yards", "proj_rush_tds",
            "proj_pass_yards", "proj_pass_tds",
        ]
        extras = [f"team_{t}" for t in TEAM_TARGETS]
        cols += extras
        print(proj[cols].round(2).to_string(index=False))

        out = input("Save CSV to artifacts/? (yes/no): ").strip().lower()
        if out in ("y", "yes"):
            path = os.path.join(ARTIFACTS_DIR, f"projections_{team}_{year}_season.csv")
            proj.to_csv(path, index=False)
            print(f"Saved: {path}")
        return

    sched = team_schedule_for_year(team, year)
    for i, r in enumerate(sched.itertuples(index=False), start=1):
        loc = "vs" if r.is_home == 1 else "@"
        print(f"{i}. Week {r.week} {loc} {r.opponent}")
    try:
        pick = int(input("Select a week number: ").strip())
    except Exception:
        print("Invalid selection.")
        sys.exit(1)
    week = int(pick)

    print("Generating week projection...")
    df, team_pred = project_team_week(year, team, week)

    phi_p, phi_r, sigma_td = _get_phi_sigma(team, year)
    print(f"Adjustments -> RZ φ_pass={phi_p:.3f}, φ_rush={phi_r:.3f}; TD σ={sigma_td:.3f}")

    print("Predicted team totals:")
    for k, v in team_pred.items():
        try:
            print(f"  {k}: {float(v):.2f}")
        except Exception:
            print(f"  {k}: {v}")
    print()

    cols = [
        "player_name", "position",
        "proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds",
        "proj_rush_att", "proj_rush_yards", "proj_rush_tds",
        "proj_pass_yards", "proj_pass_tds",
    ]
    print(df[cols].round(2).to_string(index=False))

    out = input("Save CSV to artifacts/? (yes/no): ").strip().lower()
    if out in ("y", "yes"):
        path = os.path.join(ARTIFACTS_DIR, f"projections_{team}_{year}_w{week}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    run_sim = input("Run Monte Carlo simulation from these point estimates? (yes/no): ").strip().lower()
    if run_sim in ("y", "yes"):
        draws_in = input(f"Number of draws [default {SIM_DEFAULT_DRAWS}]: ").strip()
        seed_in = input(f"Random seed [default {SIM_DEFAULT_SEED}]: ").strip()
        draws = int(draws_in) if draws_in else SIM_DEFAULT_DRAWS
        seed = int(seed_in) if seed_in else SIM_DEFAULT_SEED

        team_summary, player_summary = simulate_from_point(
            df, team_pred, draws=draws, seed=seed, team=team, year=year, week=week
        )

        print(f"Monte Carlo summary (N={draws}, seed={seed})")
        print(team_summary.round(3).to_string(index=False))
        print(player_summary.round(3).to_string(index=False))

        out_sim = input("Save simulation CSV to artifacts/? (yes/no): ").strip().lower()
        if out_sim in ("y", "yes"):
            path = os.path.join(ARTIFACTS_DIR, f"sim_{team}_{year}_w{week}_{draws}.csv")
            player_summary.to_csv(path, index=False)
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()