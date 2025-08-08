import pandas as pd
import pytest
from src.models_team import TeamProjectionModel

def test_team_model_blocks_leakage():
    df = pd.DataFrame({
        "season":[2022, 2023, 2024],
        "is_home":[1,0,1],
        "r8_pass_attempts_mean":[30,31,32],
        "r8_pass_yards_mean":[220,225,235],
        "r8_pass_tds_mean":[1.8,1.9,2.0],
        "r8_rush_attempts_mean":[22,24,23],
        "r8_rush_yards_mean":[96,101,98],
        "r8_rush_tds_mean":[0.8,0.7,0.9],
        "r8_epa_offense_mean":[0.02,0.03,0.04],
        "r8_success_offense_mean":[0.48,0.50,0.51],
        "r8_opp_allowed_epa_per_play_mean":[0.01,0.02,0.015],
        "r8_opp_allowed_success_rate_mean":[0.46,0.47,0.48],
        "pass_attempts":[34,35,36],
        "pass_yards":[250,260,270],
        "pass_tds":[2,2,3],
        "rush_attempts":[24,26,25],
        "rush_yards":[100,110,105],
        "rush_tds":[1,1,1]
    })
    model = TeamProjectionModel(projection_year=2024, seed=42)
    with pytest.raises(ValueError):
        model.fit(df)
