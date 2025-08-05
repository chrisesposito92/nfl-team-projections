"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def sample_pbp_data():
    """Sample play-by-play data for testing."""
    return pd.DataFrame({
        'game_id': ['2023_01_ARI_BUF'] * 10,
        'season': [2023] * 10,
        'week': [1] * 10,
        'season_type': ['REG'] * 10,
        'posteam': ['ARI'] * 5 + ['BUF'] * 5,
        'defteam': ['BUF'] * 5 + ['ARI'] * 5,
        'home_team': ['BUF'] * 10,
        'away_team': ['ARI'] * 10,
        'play_type': ['pass', 'pass', 'run', 'pass', 'run'] * 2,
        'yards_gained': [10, 5, 3, 15, 7, 20, 0, 4, 8, 2],
        'complete_pass': [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        'touchdown': [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        'two_point_attempt': [0] * 10
    })


@pytest.fixture
def sample_weekly_data():
    """Sample weekly player data for testing."""
    return pd.DataFrame({
        'player_id': ['P001', 'P002', 'P003', 'P004'],
        'player_display_name': ['QB Test', 'RB Test', 'WR Test', 'TE Test'],
        'position': ['QB', 'RB', 'WR', 'TE'],
        'recent_team': ['ARI', 'ARI', 'ARI', 'ARI'],
        'season': [2023] * 4,
        'week': [1] * 4,
        'targets': [0, 3, 8, 5],
        'receptions': [0, 2, 5, 4],
        'receiving_yards': [0, 15, 75, 45],
        'receiving_tds': [0, 0, 1, 0],
        'carries': [1, 15, 0, 0],
        'rushing_yards': [5, 65, 0, 0],
        'rushing_tds': [0, 1, 0, 0],
        'passing_tds': [2, 0, 0, 0]
    })


@pytest.fixture
def sample_snap_data():
    """Sample snap count data for testing."""
    return pd.DataFrame({
        'player': ['QB Test', 'RB Test', 'WR Test', 'TE Test'],
        'team': ['ARI', 'ARI', 'ARI', 'ARI'],
        'season': [2023] * 4,
        'week': [1] * 4,
        'offense_snaps': [65, 45, 55, 40],
        'offense_pct': [1.0, 0.69, 0.85, 0.62]
    })


@pytest.fixture
def sample_team_features():
    """Sample team features for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'team': ['ARI'],
        'opponent': ['BUF'],
        'season': [2023],
        'week': [1],
        'home': [0],
        'pass_attempts': [35],
        'pass_completions': [22],
        'pass_yards': [250],
        'pass_tds': [2],
        'rush_attempts': [25],
        'rush_yards': [100],
        'rush_tds': [1],
        'team_pass_attempts_avg_3g': [32.5],
        'team_pass_yards_avg_3g': [235.0],
        'team_rush_attempts_avg_3g': [27.0],
        'team_rush_yards_avg_3g': [110.0],
        'def_pass_yards_allowed_avg_3g': [220.0],
        'def_rush_yards_allowed_avg_3g': [95.0]
    })


@pytest.fixture
def sample_player_features():
    """Sample player features for testing."""
    return pd.DataFrame({
        'player_id': ['P002', 'P003', 'P004'],
        'player_display_name': ['RB Test', 'WR Test', 'TE Test'],
        'position': ['RB', 'WR', 'TE'],
        'recent_team': ['ARI', 'ARI', 'ARI'],
        'target_share': [0.1, 0.4, 0.25],
        'rush_attempt_share': [0.6, 0.0, 0.0],
        'pass_td_share': [0.0, 0.5, 0.0],
        'rush_td_share': [1.0, 0.0, 0.0],
        'targets_avg_3g': [2.5, 7.5, 4.5],
        'target_share_avg_3g': [0.08, 0.35, 0.22],
        'offense_pct': [0.69, 0.85, 0.62],
        'is_rb': [1, 0, 0],
        'is_wr': [0, 1, 0],
        'is_te': [0, 0, 1]
    })