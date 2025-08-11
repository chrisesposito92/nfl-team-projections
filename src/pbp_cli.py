# src/pbp_cli.py
from __future__ import annotations
import sys, os, argparse
import numpy as np
import pandas as pd
from .pbp import (
    build_game_priors,
    attribute_players_from_plays,
    simulate_single_game as simulate_stateful,  # stateful engine (kickoffs, OOB, etc.)
    simulate_game as simulate_simple,           # simple/pace engine
)

def main():
    parser = argparse.ArgumentParser("PBP Game Simulator (Phase-1: stateful/simple + penalties stub + soft anchor)")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--sims", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--players-csv", type=str, default=None)

    # toggles
    parser.add_argument("--engine", choices=["stateful", "simple"], default="stateful")
    parser.add_argument("--penalties", choices=["on", "off"], default="off")
    parser.add_argument("--anchor", choices=["on", "off"], default="off")
    parser.add_argument("--penalty-rate", type=float, default=0.10)   # optional
    parser.add_argument("--anchor-weight", type=float, default=0.25)  # optional

    args = parser.parse_args()

    home = args.home.upper()
    away = args.away.upper()

    # define these BEFORE use
    penalties_on = (args.penalties == "on")
    anchor_on    = (args.anchor == "on")

    priors = build_game_priors(home, away, args.year)

    if args.engine == "stateful":
        # simulate_stateful returns a single DF per call; build a list
        sims = [
            simulate_stateful(
                home, away, args.year,
                rng=np.random.RandomState(args.seed + 37 * k),
                priors=priors,
                penalties_enabled=penalties_on,
                penalty_rate=(args.penalty_rate if penalties_on else 0.0),
                apply_anchor=anchor_on,
                anchor_weight=(args.anchor_weight if anchor_on else 0.0),
            )
            for k in range(args.sims)
        ]
    else:
        # simple engine already returns a list
        sims = simulate_simple(
            home, away, args.year,
            n_sims=args.sims,
            seed=args.seed,
            priors=priors,
            penalties_enabled=penalties_on,
            penalty_rate=(args.penalty_rate if penalties_on else 0.0),
            anchor_weight=(args.anchor_weight if anchor_on else 0.0),
        )

    # First sim’s plays
    plays = sims[0]

    # Player attribution
    player_box = attribute_players_from_plays(
        plays, home=home, away=away, year=args.year, seed=args.seed or 0
    )

    # Save plays
    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        plays.to_csv(args.csv, index=False)
        print(f"Saved: {args.csv}")

    # Save player box
    players_csv_path = args.players_csv
    if players_csv_path is None and args.csv:
        root, _ = os.path.splitext(args.csv)
        players_csv_path = f"{root}_players.csv"
    if players_csv_path:
        os.makedirs(os.path.dirname(players_csv_path), exist_ok=True)
        player_box.to_csv(players_csv_path, index=False)
        print(f"Saved: {players_csv_path}")

    # Print summary for the first sim
    print(plays.tail(3).to_string(index=False))
    final_row = plays.iloc[-1]
    print(f"\nFINAL: {away} {final_row['away_score']} @ {home} {final_row['home_score']}")

if __name__ == "__main__":
    sys.exit(main())