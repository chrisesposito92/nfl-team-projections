"""Unit tests for data aggregator."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.aggregator import DataAggregator


class TestDataAggregator:
    """Test data aggregation functions."""
    
    def setup_method(self):
        """Set up test aggregator."""
        self.aggregator = DataAggregator()
    
    def test_aggregate_team_game_stats(self, sample_pbp_data):
        """Test team-level aggregation."""
        result = self.aggregator.aggregate_team_game_stats(sample_pbp_data)
        
        # Should have 2 rows (one for each team)
        assert len(result) == 2
        
        # Check ARI stats
        ari_stats = result[result['team'] == 'ARI'].iloc[0]
        assert ari_stats['pass_attempts'] == 3
        assert ari_stats['rush_attempts'] == 2
        assert ari_stats['pass_yards'] == 30
        assert ari_stats['rush_yards'] == 10
        assert ari_stats['pass_tds'] == 1
        assert ari_stats['rush_tds'] == 0
        
        # Check home/away
        assert ari_stats['home'] == 0  # ARI is away team
        
    def test_aggregate_player_game_stats(self, sample_weekly_data):
        """Test player-level aggregation with share calculations."""
        result = self.aggregator.aggregate_player_game_stats(sample_weekly_data)
        
        # Should have same number of players
        assert len(result) == len(sample_weekly_data)
        
        # Check share calculations
        total_targets = sample_weekly_data['targets'].sum()
        wr_data = result[result['position'] == 'WR'].iloc[0]
        
        expected_target_share = 8 / total_targets
        assert np.isclose(wr_data['target_share'], expected_target_share)
        
        # Check RB rush share
        total_carries = sample_weekly_data['carries'].sum()
        rb_data = result[result['position'] == 'RB'].iloc[0]
        expected_rush_share = 15 / total_carries
        assert np.isclose(rb_data['rush_attempt_share'], expected_rush_share)
    
    def test_merge_snap_data(self, sample_weekly_data, sample_snap_data):
        """Test merging snap count data."""
        # First aggregate player stats
        player_stats = self.aggregator.aggregate_player_game_stats(sample_weekly_data)
        
        # Then merge snap data
        result = self.aggregator.merge_snap_data(player_stats, sample_snap_data)
        
        # Check that snap data was merged
        assert 'offense_snaps' in result.columns
        assert 'offense_pct' in result.columns
        
        # Check specific values
        qb_data = result[result['position'] == 'QB'].iloc[0]
        assert qb_data['offense_snaps'] == 65
        assert qb_data['offense_pct'] == 1.0
    
    def test_create_training_datasets(self, sample_pbp_data, sample_weekly_data):
        """Test training dataset creation with future data prevention."""
        # Create sample data for multiple years
        team_stats = pd.DataFrame({
            'season': [2021, 2022, 2023, 2024],
            'week': [1, 1, 1, 1],
            'team': ['ARI'] * 4,
            'pass_yards': [200, 250, 300, 350]
        })
        
        player_stats = pd.DataFrame({
            'season': [2021, 2022, 2023, 2024],
            'week': [1, 1, 1, 1],
            'player_id': ['P001'] * 4,
            'targets': [5, 6, 7, 8]
        })
        
        # Create training data for 2024 projections
        result = self.aggregator.create_training_datasets(
            team_stats, player_stats, target_year=2024
        )
        
        # Should only include data before 2024
        assert result['team_train']['season'].max() == 2023
        assert result['player_train']['season'].max() == 2023
        assert len(result['team_train']) == 3
        assert len(result['player_train']) == 3