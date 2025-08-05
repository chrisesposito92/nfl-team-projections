"""Team-level projection models."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import logging
from pathlib import Path

from config import TEAM_TARGETS, TEAM_MODEL_PARAMS, RANDOM_STATE

logger = logging.getLogger(__name__)


class TeamProjectionModel:
    """Manages team-level projection models for all target stats."""
    
    def __init__(self, model_dir: Optional[str] = None):
        """Initialize team projection model.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.models = {}
        self.feature_columns = None
        self.model_dir = Path(model_dir) if model_dir else Path("models/team")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def train(
        self, 
        features: pd.DataFrame,
        targets: List[str] = TEAM_TARGETS,
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """Train models for all target statistics.
        
        Args:
            features: DataFrame with features and target columns
            targets: List of target columns to train models for
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        logger.info(f"Training team models for targets: {targets}")
        
        # Identify feature columns (exclude targets and identifiers)
        id_cols = ['game_id', 'season', 'week', 'team', 'opponent']
        self.feature_columns = [
            col for col in features.columns 
            if col not in targets + id_cols
        ]
        
        logger.info(f"Using {len(self.feature_columns)} features")
        
        # Remove any rows with NaN in features or targets
        clean_data = features.dropna(subset=self.feature_columns + targets)
        logger.info(f"Training on {len(clean_data)} samples")
        
        X = clean_data[self.feature_columns]
        
        metrics = {}
        
        for target in targets:
            logger.info(f"Training model for {target}")
            
            y = clean_data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_STATE
            )
            
            # Train model
            model = xgb.XGBRegressor(**TEAM_MODEL_PARAMS)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            self.models[target] = model
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            metrics[target] = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mean_actual': y_test.mean(),
                'mean_pred': y_pred.mean()
            }
            
            logger.info(f"{target} - MAE: {metrics[target]['mae']:.2f}, "
                       f"RMSE: {metrics[target]['rmse']:.2f}")
        
        # Save models
        self.save_models()
        
        return metrics
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for all targets.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            DataFrame with predictions for each target
        """
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Train models first.")
        
        # Ensure we have all required features
        missing_features = set(self.feature_columns) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X = features[self.feature_columns]
        predictions = pd.DataFrame(index=features.index)
        
        for target, model in self.models.items():
            predictions[f'proj_{target}'] = model.predict(X)
            
            # Ensure non-negative predictions
            predictions[f'proj_{target}'] = predictions[f'proj_{target}'].clip(lower=0)
        
        # Post-process predictions for consistency
        predictions = self._post_process_predictions(predictions)
        
        return predictions
    
    def _post_process_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Post-process predictions to ensure consistency.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Post-processed predictions
        """
        # Ensure completions <= attempts
        if 'proj_pass_completions' in predictions and 'proj_pass_attempts' in predictions:
            predictions['proj_pass_completions'] = predictions[[
                'proj_pass_completions', 'proj_pass_attempts'
            ]].min(axis=1)
        
        # Round attempts to integers
        for col in ['proj_pass_attempts', 'proj_rush_attempts']:
            if col in predictions:
                predictions[col] = predictions[col].round()
        
        # Round TDs to integers
        for col in predictions.columns:
            if 'tds' in col:
                predictions[col] = predictions[col].round()
        
        return predictions
    
    def save_models(self):
        """Save all trained models to disk."""
        for target, model in self.models.items():
            model_path = self.model_dir / f"{target}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved model for {target} to {model_path}")
        
        # Save feature columns
        feature_path = self.model_dir / "feature_columns.pkl"
        joblib.dump(self.feature_columns, feature_path)
    
    def load_models(self):
        """Load all models from disk."""
        # Load feature columns
        feature_path = self.model_dir / "feature_columns.pkl"
        if feature_path.exists():
            self.feature_columns = joblib.load(feature_path)
        
        # Load models
        for target in TEAM_TARGETS:
            model_path = self.model_dir / f"{target}_model.pkl"
            if model_path.exists():
                self.models[target] = joblib.load(model_path)
                logger.info(f"Loaded model for {target}")
    
    def get_feature_importance(self, target: str, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for a specific target model.
        
        Args:
            target: Target statistic
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if target not in self.models:
            raise ValueError(f"No model found for target: {target}")
        
        model = self.models[target]
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance