from __future__ import annotations
import sys
import pandas as pd
from .config import TEAM_TARGETS
from .data import import_team_list, team_schedule_for_year
from .predict import project_team_week, project_team_season
from .utils import set_display

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
    if full in ("y","yes"):
        print("Generating full-season projections...")
        proj = project_team_season(year, team)
        cols = [
            "season","team","week","player_name","position",
            "proj_targets","proj_receptions","proj_rec_yards","proj_rec_tds",
            "proj_rush_att","proj_rush_yards","proj_rush_tds",
            "proj_pass_yards","proj_pass_tds"
        ]
        extras = [f"team_{t}" for t in TEAM_TARGETS]
        cols += extras
        print(proj[cols].round(2).to_string(index=False))
        out = input("Save CSV to artifacts/? (yes/no): ").strip().lower()
        if out in ("y","yes"):
            path = f"artifacts/projections_{team}_{year}_season.csv"
            proj.to_csv(path, index=False)
            print(f"Saved: {path}")
        return
    # single week
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
    print("Predicted team totals:")
    for k, v in team_pred.items():
        print(f"  {k}: {v:.2f}")
    print()
    cols = [
        "player_name","position",
        "proj_targets","proj_receptions","proj_rec_yards","proj_rec_tds",
        "proj_rush_att","proj_rush_yards","proj_rush_tds",
        "proj_pass_yards","proj_pass_tds"
    ]
    print(df[cols].round(2).to_string(index=False))
    out = input("Save CSV to artifacts/? (yes/no): ").strip().lower()
    if out in ("y","yes"):
        path = f"artifacts/projections_{team}_{year}_week{week}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
