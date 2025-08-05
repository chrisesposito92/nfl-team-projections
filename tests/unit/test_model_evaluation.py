"""Model evaluation and regression tests."""

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tempfile
import shutil
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from models.team_model import TeamProjectionModel
from models.player_model import PlayerShareModel
from config import TEAM_TARGETS, PLAYER_SHARE_TARGETS


class TestModelEvaluation:
    """Test model training, evaluation, and regression prevention."""
    
    def setup_method(self):
        """Set up test models with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.team_model = TeamProjectionModel(model_dir=self.temp_dir)
        self.player_model = PlayerShareModel(model_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_team_model_training_and_metrics(self):
        """Test team model training and evaluation metrics."""
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        features = pd.DataFrame({
            'team': np.random.choice(['ARI', 'BUF', 'CAR', 'DAL'], n_samples),
            'opponent': np.random.choice(['ARI', 'BUF', 'CAR', 'DAL'], n_samples),
            'home': np.random.randint(0, 2, n_samples),
            'team_pass_attempts_avg_3g': np.random.normal(32, 5, n_samples),
            'team_pass_yards_avg_3g': np.random.normal(250, 50, n_samples),
            'team_rush_attempts_avg_3g': np.random.normal(25, 5, n_samples),
            'team_rush_yards_avg_3g': np.random.normal(100, 30, n_samples),
            'def_pass_yards_allowed_avg_3g': np.random.normal(240, 40, n_samples),
            'def_rush_yards_allowed_avg_3g': np.random.normal(110, 25, n_samples),
        })
        
        # Create correlated targets based on features
        features['pass_attempts'] = (
            features['team_pass_attempts_avg_3g'] + 
            np.random.normal(0, 3, n_samples)
        ).clip(lower=0)
        
        features['pass_yards'] = (
            features['team_pass_yards_avg_3g'] * 0.8 +
            features['def_pass_yards_allowed_avg_3g'] * 0.2 +
            np.random.normal(0, 20, n_samples)
        ).clip(lower=0)
        
        features['rush_attempts'] = (
            features['team_rush_attempts_avg_3g'] +
            np.random.normal(0, 3, n_samples)
        ).clip(lower=0)
        
        features['rush_yards'] = (
            features['team_rush_yards_avg_3g'] * 0.7 +
            features['def_rush_yards_allowed_avg_3g'] * 0.3 +
            np.random.normal(0, 15, n_samples)
        ).clip(lower=0)
        
        # Add other targets
        features['pass_completions'] = (features['pass_attempts'] * 0.65).clip(lower=0)
        features['pass_tds'] = (features['pass_attempts'] * 0.06).clip(lower=0)
        features['rush_tds'] = (features['rush_attempts'] * 0.04).clip(lower=0)
        
        # Train models
        metrics = self.team_model.train(features, test_size=0.2)
        
        # Verify metrics structure
        assert len(metrics) == len(TEAM_TARGETS)
        for target in TEAM_TARGETS:
            assert target in metrics
            assert 'mae' in metrics[target]
            assert 'rmse' in metrics[target]
            assert 'mean_actual' in metrics[target]
            assert 'mean_pred' in metrics[target]
        
        # Check that predictions are reasonable
        # MAE should be less than 50% of mean actual for most stats
        for target in ['pass_yards', 'rush_yards']:
            mae_ratio = metrics[target]['mae'] / metrics[target]['mean_actual']
            assert mae_ratio < 0.5, f"MAE ratio for {target} is too high: {mae_ratio}"
        
        # Test feature importance
        importance = self.team_model.get_feature_importance('pass_yards', top_n=5)
        assert len(importance) == 5
        assert importance.iloc[0]['importance'] > 0  # Top feature should have importance
    
    def test_player_model_share_constraints(self):
        """Test player model respects share constraints."""
        # Create synthetic player data
        np.random.seed(42)
        n_samples = 500
        
        positions = np.random.choice(['RB', 'WR', 'TE'], n_samples)
        features = pd.DataFrame({
            'player_id': [f'P{i}' for i in range(n_samples)],
            'player_display_name': [f'Player {i}' for i in range(n_samples)],
            'position': positions,  # Keep for later but not for training
            'recent_team': np.random.choice(['ARI', 'BUF', 'CIN'], n_samples),
            'is_rb': [1 if p == 'RB' else 0 for p in positions],
            'is_wr': [1 if p == 'WR' else 0 for p in positions],
            'is_te': [1 if p == 'TE' else 0 for p in positions],
            'targets_avg_3g': np.random.uniform(0, 10, n_samples),
            'target_share_avg_3g': np.random.uniform(0, 0.3, n_samples),
            'offense_pct': np.random.uniform(0.3, 1.0, n_samples),
            'seasons_played': np.random.randint(0, 10, n_samples),
        })
        
        # Create realistic target shares
        features['target_share'] = (
            features['target_share_avg_3g'] * 0.8 +
            np.random.normal(0, 0.05, n_samples)
        ).clip(0, 0.5)
        
        features['rush_attempt_share'] = np.where(
            features['is_rb'] == 1,
            np.random.uniform(0.2, 0.8, n_samples),
            np.random.uniform(0, 0.05, n_samples)
        )
        
        features['pass_td_share'] = features['target_share'] * np.random.uniform(0.8, 1.2, n_samples)
        features['rush_td_share'] = features['rush_attempt_share'] * np.random.uniform(0.8, 1.2, n_samples)
        
        # Train model
        metrics = self.player_model.train(features, test_size=0.2)
        
        # Check all shares are predicted
        for share in PLAYER_SHARE_TARGETS:
            assert share in metrics
            assert metrics[share]['mae'] < 0.2  # Shares should have low MAE
        
        # Test predictions stay within bounds
        test_features = features.iloc[:10].copy()
        predictions = self.player_model.predict(test_features)
        
        for share_col in [col for col in predictions.columns if 'proj_' in col and 'share' in col]:
            assert predictions[share_col].min() >= 0
            assert predictions[share_col].max() <= 1
    
    def test_model_regression_prevention(self):
        """Test that model performance doesn't regress with updates."""
        # Create benchmark dataset
        np.random.seed(42)
        benchmark_data = self._create_benchmark_data()
        
        # Train initial model
        initial_metrics = self.team_model.train(
            benchmark_data,
            targets=['pass_yards', 'rush_yards'],  # Test subset
            test_size=0.2
        )
        
        # Save initial performance
        initial_pass_mae = initial_metrics['pass_yards']['mae']
        initial_rush_mae = initial_metrics['rush_yards']['mae']
        
        # Simulate model update with same data (should get similar performance)
        # Reset model
        self.team_model = TeamProjectionModel(model_dir=self.temp_dir)
        
        # Retrain
        updated_metrics = self.team_model.train(
            benchmark_data,
            targets=['pass_yards', 'rush_yards'],
            test_size=0.2
        )
        
        # Performance shouldn't degrade by more than 10%
        pass_mae_change = abs(updated_metrics['pass_yards']['mae'] - initial_pass_mae) / initial_pass_mae
        rush_mae_change = abs(updated_metrics['rush_yards']['mae'] - initial_rush_mae) / initial_rush_mae
        
        assert pass_mae_change < 0.1, f"Pass yards MAE changed by {pass_mae_change:.1%}"
        assert rush_mae_change < 0.1, f"Rush yards MAE changed by {rush_mae_change:.1%}"
    
    def test_model_save_and_load(self):
        """Test model persistence."""
        # Create and train model
        features = self._create_benchmark_data()
        self.team_model.train(features, targets=['pass_yards'], test_size=0.2)
        
        # Make predictions with original model
        test_features = features.iloc[:5]
        original_predictions = self.team_model.predict(test_features)
        
        # Create new model instance and load
        new_model = TeamProjectionModel(model_dir=self.temp_dir)
        new_model.load_models()
        
        # Make predictions with loaded model
        loaded_predictions = new_model.predict(test_features)
        
        # Predictions should be identical
        pd.testing.assert_frame_equal(original_predictions, loaded_predictions)
    
    def test_evaluation_on_holdout_season(self):
        """Test model evaluation on a held-out season."""
        # Create multi-season data
        seasons_data = []
        for season in [2021, 2022, 2023, 2024]:
            season_features = self._create_benchmark_data()
            season_features['season'] = season
            seasons_data.append(season_features)
        
        all_data = pd.concat(seasons_data, ignore_index=True)
        
        # Train on 2021-2023, test on 2024
        train_data = all_data[all_data['season'] < 2024]
        test_data = all_data[all_data['season'] == 2024]
        
        # Train model
        self.team_model.train(train_data, targets=['pass_yards', 'rush_yards'])
        
        # Evaluate on holdout season
        predictions = self.team_model.predict(test_data)
        
        # Calculate metrics
        pass_mae = mean_absolute_error(
            test_data['pass_yards'],
            predictions['proj_pass_yards']
        )
        
        rush_mae = mean_absolute_error(
            test_data['rush_yards'],
            predictions['proj_rush_yards']
        )
        
        # Performance should be reasonable
        assert pass_mae < test_data['pass_yards'].mean() * 0.3
        assert rush_mae < test_data['rush_yards'].mean() * 0.3
    
    def _create_benchmark_data(self, n_samples=500):
        """Create standardized benchmark dataset for testing."""
        np.random.seed(42)
        
        features = pd.DataFrame({
            'team': np.random.choice(['ARI', 'BUF', 'CAR', 'DAL'], n_samples),
            'opponent': np.random.choice(['ARI', 'BUF', 'CAR', 'DAL'], n_samples),
            'home': np.random.randint(0, 2, n_samples),
            'team_pass_attempts_avg_3g': np.random.normal(32, 5, n_samples),
            'team_pass_yards_avg_3g': np.random.normal(250, 50, n_samples),
            'team_rush_attempts_avg_3g': np.random.normal(25, 5, n_samples),
            'team_rush_yards_avg_3g': np.random.normal(100, 30, n_samples),
            'def_pass_yards_allowed_avg_3g': np.random.normal(240, 40, n_samples),
            'def_rush_yards_allowed_avg_3g': np.random.normal(110, 25, n_samples),
        })
        
        # Create realistic correlated targets
        features['pass_yards'] = (
            features['team_pass_yards_avg_3g'] * 0.9 +
            features['def_pass_yards_allowed_avg_3g'] * 0.1 +
            np.random.normal(0, 25, n_samples)
        ).clip(lower=0)
        
        features['rush_yards'] = (
            features['team_rush_yards_avg_3g'] * 0.85 +
            features['def_rush_yards_allowed_avg_3g'] * 0.15 +
            np.random.normal(0, 20, n_samples)
        ).clip(lower=0)
        
        return features