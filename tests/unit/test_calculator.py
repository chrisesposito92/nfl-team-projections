"""Unit tests for projection calculator."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from projections.calculator import ProjectionCalculator


class TestProjectionCalculator:
    """Test projection calculation functions."""
    
    def setup_method(self):
        """Set up test calculator."""
        self.calculator = ProjectionCalculator()
    
    def test_calculate_player_projections(self):
        """Test player projection calculations."""
        # Mock team projections
        team_projections = pd.DataFrame({
            'proj_pass_attempts': [35],
            'proj_pass_completions': [22],
            'proj_pass_yards': [275],
            'proj_pass_tds': [2],
            'proj_rush_attempts': [25],
            'proj_rush_yards': [110],
            'proj_rush_tds': [1]
        })
        
        # Mock player shares
        player_shares = pd.DataFrame({
            'player_id': ['P001', 'P002', 'P003'],
            'player_display_name': ['RB Test', 'WR Test', 'TE Test'],
            'position': ['RB', 'WR', 'TE'],
            'recent_team': ['ARI', 'ARI', 'ARI'],
            'proj_target_share': [0.15, 0.40, 0.25],
            'proj_rush_attempt_share': [0.60, 0.0, 0.0],
            'proj_pass_td_share': [0.0, 0.5, 0.25],
            'proj_rush_td_share': [1.0, 0.0, 0.0]
        })
        
        # Mock efficiency metrics
        player_efficiency = pd.DataFrame({
            'player_id': ['P001', 'P002', 'P003'],
            'catch_rate': [0.7, 0.65, 0.75],
            'yards_per_target': [6.0, 10.0, 8.5],
            'yards_per_carry': [4.5, 0.0, 0.0]
        })
        
        result = self.calculator.calculate_player_projections(
            team_projections, player_shares, player_efficiency
        )
        
        # Check calculations
        rb_proj = result[result['position'] == 'RB'].iloc[0]
        wr_proj = result[result['position'] == 'WR'].iloc[0]
        
        # RB should have targets and rushing attempts
        assert rb_proj['proj_targets'] == round(35 * 0.15)
        assert rb_proj['proj_rush_attempts'] == round(25 * 0.60)
        assert rb_proj['proj_rushing_tds'] == 1
        
        # WR should have targets but no rushing
        assert wr_proj['proj_targets'] == round(35 * 0.40)
        assert wr_proj['proj_rush_attempts'] == 0
        assert wr_proj['proj_receiving_tds'] == 1
    
    def test_clean_projections(self):
        """Test projection cleaning and validation."""
        # Create projections with issues
        projections = pd.DataFrame({
            'player_id': ['P001', 'P002'],
            'position': ['RB', 'WR'],
            'proj_targets': [10, 15],
            'proj_receptions': [12, 10],  # More receptions than targets for RB
            'proj_pass_attempts': [5, 10],  # Non-QB with passing stats
            'proj_passing_yards': [50, 100],
            'proj_receiving_yards': [-10, 150]  # Negative yards
        })
        
        cleaned = self.calculator._clean_projections(projections)
        
        # Receptions should be capped at targets
        assert cleaned.iloc[0]['proj_receptions'] == 10
        
        # Non-QB shouldn't have passing stats
        assert cleaned.iloc[0]['proj_pass_attempts'] == 0
        assert cleaned.iloc[1]['proj_pass_attempts'] == 0
        
        # No negative values
        assert cleaned.iloc[0]['proj_receiving_yards'] == 0
    
    def test_aggregate_season_projections(self):
        """Test aggregating weekly projections to season totals."""
        # Create 3 weeks of projections
        week1 = pd.DataFrame({
            'player_id': ['P001', 'P002'],
            'player_display_name': ['Player 1', 'Player 2'],
            'position': ['RB', 'WR'],
            'recent_team': ['ARI', 'ARI'],
            'proj_rushing_yards': [50, 0],
            'proj_receiving_yards': [20, 80]
        })
        
        week2 = week1.copy()
        week2['proj_rushing_yards'] = [60, 0]
        week2['proj_receiving_yards'] = [15, 90]
        
        week3 = week1.copy()
        week3['proj_rushing_yards'] = [40, 0]
        week3['proj_receiving_yards'] = [25, 75]
        
        result = self.calculator.aggregate_season_projections([week1, week2, week3])
        
        # Check totals
        rb_total = result[result['position'] == 'RB'].iloc[0]
        assert rb_total['proj_rushing_yards'] == 150  # 50 + 60 + 40
        assert rb_total['proj_receiving_yards'] == 60  # 20 + 15 + 25
        
        wr_total = result[result['position'] == 'WR'].iloc[0]
        assert wr_total['proj_receiving_yards'] == 245  # 80 + 90 + 75
    
    def test_format_output(self):
        """Test output formatting."""
        projections = pd.DataFrame({
            'player_id': ['P001', 'P002'],
            'player_display_name': ['RB Test', 'WR Test'],
            'position': ['RB', 'WR'],
            'recent_team': ['ARI', 'ARI'],
            'proj_pass_attempts': [0, 0],
            'proj_pass_completions': [0, 0],
            'proj_passing_yards': [0, 0],
            'proj_passing_tds': [0, 0],
            'proj_targets': [5, 10],
            'proj_receptions': [4, 7],
            'proj_receiving_yards': [30, 100],
            'proj_receiving_tds': [0, 1],
            'proj_rush_attempts': [15, 0],
            'proj_rushing_yards': [60, 0],
            'proj_rushing_tds': [1, 0]
        })
        
        formatted = self.calculator.format_output(projections, week=1)
        
        # Check column renaming
        assert 'Player' in formatted.columns
        assert 'Pos' in formatted.columns
        assert 'Rush Yds' in formatted.columns
        
        # Check week was added
        assert 'Week' in formatted.columns
        assert formatted['Week'].iloc[0] == 1
        
        # Check sorting (WR with more fantasy points should be second)
        assert formatted.iloc[0]['Pos'] == 'RB'
        assert formatted.iloc[1]['Pos'] == 'WR'