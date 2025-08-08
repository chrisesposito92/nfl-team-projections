from __future__ import annotations
import sys
import os
import pandas as pd
from .config import TEAM_TARGETS, SIM_DEFAULT_DRAWS, SIM_DEFAULT_SEED, ARTIFACTS_DIR
from .data import import_team_list, team_schedule_for_year
from .predict import project_team_week, project_team_season
from .utils import set_display
from .simulate import simulate_from_point
from .simulate_calibrated import simulate_from_point_calibrated

def main():
    set_display()
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
        proj = project_team_season(year, team)
        cols = [
            "season","team","week","player_name","position",
            "proj_targets","proj_receptions","proj_rec_yards","proj_rec_tds",
            "proj_rush_att","proj_rush_yards","proj_rush_tds",
            "proj_pass_yards","proj_pass_tds"
        ]
        extras = [f"team_{t}" for t in TEAM_TARGETS]
        cols = [c for c in cols + extras if c in proj.columns]
        print(proj[cols].round(2).to_string(index=False))
        out = input("Save CSV to artifacts/? (yes/no): ").strip().lower()
        if out in ("y", "yes"):
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            path = os.path.join(ARTIFACTS_DIR, f"projections_{team}_{year}_season.csv")
            proj.to_csv(path, index=False)
            print(f"Saved: {path}")
        return

    sched = team_schedule_for_year(team, year)
    for i, r in enumerate(sched.itertuples(index=False), start=1):
        loc = "vs" if getattr(r, "is_home", 0) == 1 else "@"
        print(f"{i}. Week {getattr(r, 'week', i)} {loc} {getattr(r, 'opponent', '')}")

    try:
        pick = int(input("Select a week number: ").strip())
    except Exception:
        print("Invalid selection.")
        sys.exit(1)
    if pick < 1 or pick > len(sched):
        print("Invalid selection.")
        sys.exit(1)

    if "week" in sched.columns:
        week = int(sched.iloc[pick - 1]["week"])
    else:
        week = pick

    print("Generating week projection...")
    df, team_pred = project_team_week(year, team, week)

    print("Predicted team totals:")
    for k, v in team_pred.items():
        try:
            print(f"  {k}: {float(v):.2f}")
        except Exception:
            print(f"  {k}: {v}")
    print()

    cols = [
        "player_name","position",
        "proj_targets","proj_receptions","proj_rec_yards","proj_rec_tds",
        "proj_rush_att","proj_rush_yards","proj_rush_tds",
        "proj_pass_yards","proj_pass_tds"
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].round(2).to_string(index=False))

    out = input("Save CSV to artifacts/? (yes/no): ").strip().lower()
    if out in ("y", "yes"):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        path = os.path.join(ARTIFACTS_DIR, f"projections_{team}_{year}_week{week}.csv")
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    ans = input("Run Monte Carlo simulation? (yes/no): ").strip().lower()
    if ans in ("y","yes"):
        mode = input("Use calibrated Option B if available? (yes/no): ").strip().lower()
        s = input(f"Number of draws [{SIM_DEFAULT_DRAWS}]: ").strip()
        n_draws = int(s) if s else SIM_DEFAULT_DRAWS
        t = input(f"Seed [{SIM_DEFAULT_SEED}]: ").strip()
        sim_seed = int(t) if t else SIM_DEFAULT_SEED

        if mode in ("y","yes"):
            sim_players, sim_team = simulate_from_point_calibrated(df, team_pred, team, year, n_draws, sim_seed)
        else:
            from .simulate import simulate_from_point
            sim_players, sim_team = simulate_from_point(df, team_pred, team, year, n_draws, sim_seed)

        print(f"Monte Carlo summary (N={n_draws}, seed={sim_seed})")
        print(sim_team.to_string(index=False))

        sim_cols = [
            "player_name","position",
            "targets_p50","receptions_p50","rec_yards_p50","rec_tds_p50",
            "rush_att_p50","rush_yards_p50","rush_tds_p50",
            "pass_yards_p50","pass_tds_p50"
        ]
        sim_cols = [c for c in sim_cols if c in sim_players.columns]
        print(sim_players[sim_cols].round(2).to_string(index=False))

        sv = input("Save simulation CSV to artifacts/? (yes/no): ").strip().lower()
        if sv in ("y","yes"):
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            sim_path = os.path.join(ARTIFACTS_DIR, f"sim_{team}_{year}_w{week}_{n_draws}.csv")
            sim_players.to_csv(sim_path, index=False)
            print(f"Saved: {sim_path}")

if __name__ == "__main__":
    main()