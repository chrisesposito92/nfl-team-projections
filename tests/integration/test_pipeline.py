"""Integration tests for end-to-end pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from data.loader import NFLDataLoader
from data.aggregator import DataAggregator
from features.team_features import TeamFeatureEngineer
from features.player_features import PlayerFeatureEngineer
from models.team_model import TeamProjectionModel
from models.player_model import PlayerShareModel
from projections.calculator import ProjectionCalculator


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test the complete projection pipeline."""
    
    def setup_method(self):
        """Set up pipeline components."""
        self.loader = NFLDataLoader()
        self.aggregator = DataAggregator()
        self.team_feature_engineer = TeamFeatureEngineer()
        self.player_feature_engineer = PlayerFeatureEngineer()
        self.team_model = TeamProjectionModel()
        self.player_model = PlayerShareModel()
        self.calculator = ProjectionCalculator()
    
    @patch('nfl_data_py.import_pbp_data')
    @patch('nfl_data_py.import_weekly_data')
    @patch('nfl_data_py.import_snap_counts')
    @patch('nfl_data_py.import_schedules')
    @patch('nfl_data_py.import_weekly_rosters')
    @patch('nfl_data_py.import_injuries')
    @patch('nfl_data_py.import_team_desc')
    def test_full_pipeline_single_week(
        self, mock_teams, mock_injuries, mock_rosters, mock_schedules,
        mock_snaps, mock_weekly, mock_pbp
    ):
        """Test full pipeline for single week projection."""
        # Create comprehensive mock data
        mock_pbp.return_value = self._create_mock_pbp_data()
        mock_weekly.return_value = self._create_mock_weekly_data()
        mock_snaps.return_value = self._create_mock_snap_data()
        mock_schedules.return_value = self._create_mock_schedule_data()
        mock_rosters.return_value = self._create_mock_roster_data()
        mock_injuries.return_value = pd.DataFrame()  # No injuries
        mock_teams.return_value = pd.DataFrame({
            'team_abbr': ['ARI', 'BUF'],
            'team_name': ['Arizona Cardinals', 'Buffalo Bills']
        })
        
        # Load and aggregate data
        years = [2022, 2023]
        data = self.loader.load_all_data(years)
        
        team_stats = self.aggregator.aggregate_team_game_stats(data['pbp'])
        player_stats = self.aggregator.aggregate_player_game_stats(data['weekly'])
        player_stats = self.aggregator.merge_snap_data(player_stats, data['snaps'])
        
        # Create training datasets
        datasets = self.aggregator.create_training_datasets(
            team_stats, player_stats, target_year=2024
        )
        
        # Engineer features
        team_features = self.team_feature_engineer.create_features(datasets['team_train'])
        player_features = self.player_feature_engineer.create_features(datasets['player_train'])
        
        # Train models (with minimal data)
        assert len(team_features) > 0
        assert len(player_features) > 0
        
        # Verify pipeline components work together
        assert 'team_pass_attempts_avg_3g' in team_features.columns
        assert 'target_share_avg_3g' in player_features.columns
    
    def test_projection_calculation_integration(self):
        """Test integration between models and calculator."""
        # Create mock team projections
        team_projections = pd.DataFrame({
            'proj_pass_attempts': [35],
            'proj_pass_completions': [22],
            'proj_pass_yards': [275],
            'proj_pass_tds': [2],
            'proj_rush_attempts': [25],
            'proj_rush_yards': [110],
            'proj_rush_tds': [1]
        })
        
        # Create mock player shares for a full team
        player_shares = pd.DataFrame({
            'player_id': ['QB1', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE1'],
            'player_display_name': ['QB Test', 'RB1 Test', 'RB2 Test', 
                                   'WR1 Test', 'WR2 Test', 'WR3 Test', 'TE Test'],
            'position': ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE'],
            'recent_team': ['ARI'] * 7,
            'proj_target_share': [0.0, 0.10, 0.05, 0.30, 0.25, 0.15, 0.15],
            'proj_rush_attempt_share': [0.05, 0.60, 0.30, 0.02, 0.02, 0.01, 0.0],
            'proj_pass_td_share': [0.0, 0.0, 0.0, 0.35, 0.30, 0.20, 0.15],
            'proj_rush_td_share': [0.0, 0.70, 0.30, 0.0, 0.0, 0.0, 0.0]
        })
        
        # Normalize shares
        player_shares = self.player_model.normalize_team_shares(player_shares, 'ARI')
        
        # Verify shares sum to 1.0
        assert np.isclose(player_shares['proj_target_share'].sum(), 1.0)
        assert np.isclose(player_shares['proj_rush_attempt_share'].sum(), 1.0)
        
        # Calculate projections
        player_efficiency = pd.DataFrame()  # Will use defaults
        projections = self.calculator.calculate_player_projections(
            team_projections, player_shares, player_efficiency
        )
        
        # Verify calculations
        assert len(projections) == 7  # All players
        
        # Check QB gets passing stats
        qb_proj = projections[projections['position'] == 'QB'].iloc[0]
        assert qb_proj['proj_pass_attempts'] == 35
        assert qb_proj['proj_passing_yards'] == 275
        assert qb_proj['proj_passing_tds'] == 2
        
        # Check total targets approximately equal team pass attempts
        total_targets = projections['proj_targets'].sum()
        assert abs(total_targets - 35) <= 1  # Allow for rounding
        
        # Check rushing attempts distributed correctly
        rb1_proj = projections[projections['player_display_name'] == 'RB1 Test'].iloc[0]
        assert rb1_proj['proj_rush_attempts'] == round(25 * 0.60)
    
    def test_season_aggregation_integration(self):
        """Test aggregating multiple weeks to season totals."""
        weekly_projections = []
        
        # Create 3 weeks of projections
        for week in range(1, 4):
            week_proj = pd.DataFrame({
                'player_id': ['P001', 'P002'],
                'player_display_name': ['RB Test', 'WR Test'],
                'position': ['RB', 'WR'],
                'recent_team': ['ARI', 'ARI'],
                'proj_pass_attempts': [0, 0],
                'proj_pass_completions': [0, 0],
                'proj_passing_yards': [0, 0],
                'proj_passing_tds': [0, 0],
                'proj_targets': [4, 8],
                'proj_receptions': [3, 6],
                'proj_receiving_yards': [25, 85],
                'proj_receiving_tds': [0, 1] if week == 2 else [0, 0],
                'proj_rush_attempts': [15, 1],
                'proj_rushing_yards': [65, 5],
                'proj_rushing_tds': [1, 0] if week == 1 else [0, 0]
            })
            weekly_projections.append(week_proj)
        
        # Aggregate season
        season_totals = self.calculator.aggregate_season_projections(weekly_projections)
        
        # Verify totals
        rb_season = season_totals[season_totals['position'] == 'RB'].iloc[0]
        assert rb_season['proj_rush_attempts'] == 45  # 15 * 3
        assert rb_season['proj_rushing_tds'] == 1  # Only week 1
        
        wr_season = season_totals[season_totals['position'] == 'WR'].iloc[0]
        assert wr_season['proj_targets'] == 24  # 8 * 3
        assert wr_season['proj_receiving_tds'] == 1  # Only week 2
    
    def _create_mock_pbp_data(self):
        """Create mock play-by-play data for testing."""
        games = []
        teams = ['ARI', 'BUF', 'CAR', 'DAL']
        
        for season in [2022, 2023]:
            for week in range(1, 5):
                for i in range(0, len(teams), 2):
                    game_id = f"{season}_{week:02d}_{teams[i]}_{teams[i+1]}"
                    
                    # Create plays for both teams
                    for team_idx in range(2):
                        posteam = teams[i + team_idx]
                        defteam = teams[i + 1 - team_idx]
                        
                        for play_num in range(30):  # 30 plays per team
                            play_type = 'pass' if play_num % 3 != 2 else 'run'
                            yards = np.random.randint(-5, 20)
                            
                            games.append({
                                'game_id': game_id,
                                'season': season,
                                'week': week,
                                'season_type': 'REG',
                                'posteam': posteam,
                                'defteam': defteam,
                                'home_team': teams[i+1],
                                'away_team': teams[i],
                                'play_type': play_type,
                                'yards_gained': max(0, yards),
                                'complete_pass': 1 if play_type == 'pass' and np.random.random() > 0.35 else 0,
                                'touchdown': 1 if np.random.random() < 0.05 else 0,
                                'two_point_attempt': 0
                            })
        
        return pd.DataFrame(games)
    
    def _create_mock_weekly_data(self):
        """Create mock weekly player data."""
        players = []
        positions = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE']
        
        for season in [2022, 2023]:
            for week in range(1, 5):
                for team in ['ARI', 'BUF']:
                    for pos_idx, pos in enumerate(positions):
                        player_id = f"{team}_{pos}_{pos_idx}"
                        
                        if pos == 'QB':
                            targets, carries = 0, 2
                            passing_tds = np.random.randint(0, 3)
                        elif pos == 'RB':
                            targets = np.random.randint(2, 6)
                            carries = np.random.randint(10, 20)
                            passing_tds = 0
                        else:  # WR/TE
                            targets = np.random.randint(4, 10)
                            carries = 0
                            passing_tds = 0
                        
                        players.append({
                            'player_id': player_id,
                            'player_display_name': f"{pos} {team} {pos_idx}",
                            'position': pos,
                            'recent_team': team,
                            'season': season,
                            'week': week,
                            'targets': targets,
                            'receptions': int(targets * 0.65),
                            'receiving_yards': targets * np.random.randint(6, 12),
                            'receiving_tds': 1 if np.random.random() < 0.1 else 0,
                            'carries': carries,
                            'rushing_yards': carries * np.random.randint(3, 6),
                            'rushing_tds': 1 if np.random.random() < 0.05 else 0,
                            'passing_tds': passing_tds
                        })
        
        return pd.DataFrame(players)
    
    def _create_mock_snap_data(self):
        """Create mock snap count data."""
        return pd.DataFrame({
            'player': ['QB ARI 0', 'RB ARI 1', 'WR ARI 3'],
            'team': ['ARI', 'ARI', 'ARI'],
            'season': [2023, 2023, 2023],
            'week': [1, 1, 1],
            'offense_snaps': [65, 45, 55],
            'offense_pct': [1.0, 0.69, 0.85]
        })
    
    def _create_mock_schedule_data(self):
        """Create mock schedule data."""
        return pd.DataFrame({
            'season': [2024] * 4,
            'week': [1, 2, 3, 4],
            'home_team': ['BUF', 'ARI', 'ARI', 'CAR'],
            'away_team': ['ARI', 'SEA', 'LAR', 'ARI']
        })
    
    def _create_mock_roster_data(self):
        """Create mock roster data."""
        rosters = []
        for week in range(1, 5):
            for pos_idx, pos in enumerate(['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE']):
                rosters.append({
                    'season': 2024,
                    'week': week,
                    'team': 'ARI',
                    'player_id': f"ARI_{pos}_{pos_idx}",
                    'player_display_name': f"{pos} Test {pos_idx}",
                    'position': pos
                })
        return pd.DataFrame(rosters)