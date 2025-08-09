# src/pbp_cli.py
from __future__ import annotations
import sys, os, argparse
import pandas as pd
from .pbp import simulate_game, build_game_priors, attribute_players_from_plays

def main():
    parser = argparse.ArgumentParser("PBP Game Simulator (Phase-1 complete: penalties stub + soft anchor)")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--sims", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--players-csv", type=str, default=None)

    # NEW toggles
    parser.add_argument("--penalties", action="store_true", help="Enable placeholder penalties (annotation only).")
    parser.add_argument("--penalty-rate", type=float, default=0.0, help="Per-play penalty chance (0..1). Default 0.")
    parser.add_argument("--anchor-weight", type=float, default=0.0, help="Soft-anchoring weight for pass/rush yards (0..1). Default 0 (off).")

    args = parser.parse_args()

    priors = build_game_priors(args.home.upper(), args.away.upper(), args.year)
    sims = simulate_game(
        args.home.upper(),
        args.away.upper(),
        args.year,
        n_sims=args.sims,
        seed=args.seed,
        priors=priors,
        penalties_enabled=args.penalties,
        penalty_rate=args.penalty_rate,
        anchor_weight=args.anchor_weight,
    )

    # First sim’s plays
    plays = sims[0]

    # Player attribution using NGS tilts + base shares (already in your codebase)
    player_box = attribute_players_from_plays(
        plays,
        home=args.home.upper(),
        away=args.away.upper(),
        year=args.year,
        seed=args.seed or 0,
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
    df = plays
    print(df.tail(3).to_string(index=False))
    home = args.home.upper()
    away = args.away.upper()
    final_row = df.iloc[-1]
    print(f"\nFINAL: {away} {final_row['away_score']} @ {home} {final_row['home_score']}")

if __name__ == "__main__":
    sys.exit(main())