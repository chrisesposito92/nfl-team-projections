# NFL Offensive Projections Modeler — V1.0

This repository contains a complete, deterministic first version of the **NFL Offensive Projections Modeler** described in the PRD. It trains two modeling layers:

1. **Team-Level Models** (Ridge regressions per target) to project team totals: `pass_attempts, pass_yards, pass_tds, rush_attempts, rush_yards, rush_tds`.
2. **Player-Share Models** (Gradient Boosting) to project each active player's target and rushing-attempt shares. Shares are normalized per team-week. Player stat-lines are then derived deterministically from team totals and player efficiency rates computed from historical data.

The CLI guides the user to select **year**, **team**, and **timeframe** (single week or full regular season), then prints a clean table of projections.

> Data source: `nfl_data_py` exclusively, per PRD V1.0 scope.


## Quickstart

1. Python 3.10+ recommended.
2. Install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the CLI:
   ```bash
   python -m src.cli
   ```

The first run will download data from the nflverse and train models using **seasons strictly prior to your projection year** to prevent future data leakage. Trained models and outputs are cached under `artifacts/`.

## Notes

- V1 assumes **regular season** only and positions **QB, RB, WR, TE**.
- Rookies or players without NFL history fall back to **position-level averages** and generally receive minimal workload by design.
- The full-season run uses preseason rolling context; it does **not** autoregress updates week-to-week in V1.
- Determinism: fixed random seeds, stable sorting, and no stochastic estimators in the team layer.

## Project Layout

```
nfl_offensive_projections/
├─ artifacts/                    # cached models and output files
├─ src/
│  ├─ cli.py                     # interactive CLI
│  ├─ config.py                  # constants and seeds
│  ├─ data.py                    # data loading (nfl_data_py)
│  ├─ features.py                # feature builders for team + player layers
│  ├─ models_team.py             # team-level model class
│  ├─ models_shares.py           # player-share model class
│  ├─ predict.py                 # end-to-end orchestration for projections
│  └─ utils.py                   # helpers (determinism, normalization, IO)
├─ tests/
│  ├─ test_leakage.py
│  ├─ test_normalize.py
│  └─ test_stats_math.py
└─ requirements.txt
```

## Example

```
$ python -m src.cli
NFL Offensive Projections Modeler (V1.0)
Enter projection year (e.g., 2025): 2025
Valid teams: ARI, ATL, BAL, BUF, ...
Select a team abbreviation (e.g., CIN): CIN
Project full REGULAR season? (yes/no): no
1. Week 1 vs KC
2. Week 2 @ CLE
...
Select a week number: 1

...training (first run only)...

Final player projections for CIN, Week 1:
<tabular output>
```

## Testing

```
pytest -q
```

The tests are fast, synthetic checks that verify:
- share normalization sums to 1.0 per grouping,
- no leakage from the projection year into training,
- stat allocation sums match team totals.

---

**PRD Reference:** See the attached PRD for goals, scope, requirements, and constraints.
