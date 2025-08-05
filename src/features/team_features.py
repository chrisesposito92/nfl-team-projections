"""Feature engineering for team-level projections."""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TeamFeatureEngineer:
    """Creates features for team-level projection models."""
    
    def __init__(self, rolling_windows: List[int] = [3, 6, 12]):
        """Initialize feature engineer.
        
        Args:
            rolling_windows: List of game windows for rolling statistics
        """
        self.rolling_windows = rolling_windows
    
    def create_features(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Create all features for team projections.
        
        Args:
            team_stats: DataFrame with team-game statistics
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating team features")
        
        # Sort by team and date
        team_stats = team_stats.sort_values(['team', 'season', 'week'])
        
        # Create basic features
        features = self._create_basic_features(team_stats)
        
        # Create rolling features for offense
        features = self._create_rolling_features(features, 'team')
        
        # Create opponent rolling features
        features = self._create_opponent_features(features)
        
        # Create efficiency metrics
        features = self._create_efficiency_features(features)
        
        # Create game context features
        features = self._create_context_features(features)
        
        logger.info(f"Created {len(features.columns)} features")
        
        return features
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features from raw stats.
        
        Args:
            df: Team statistics DataFrame
            
        Returns:
            DataFrame with basic features
        """
        df = df.copy()
        
        # Pass/run ratio
        df['pass_rate'] = df['pass_attempts'] / (df['pass_attempts'] + df['rush_attempts'])
        df['pass_rate'] = df['pass_rate'].fillna(0.5)  # Default to 50/50 if no plays
        
        # Completion rate
        df['completion_rate'] = np.where(
            df['pass_attempts'] > 0,
            df['pass_completions'] / df['pass_attempts'],
            0
        )
        
        # Yards per attempt
        df['yards_per_pass'] = np.where(
            df['pass_attempts'] > 0,
            df['pass_yards'] / df['pass_attempts'],
            0
        )
        
        df['yards_per_rush'] = np.where(
            df['rush_attempts'] > 0,
            df['rush_yards'] / df['rush_attempts'],
            0
        )
        
        # TD rates
        df['pass_td_rate'] = np.where(
            df['pass_attempts'] > 0,
            df['pass_tds'] / df['pass_attempts'],
            0
        )
        
        df['rush_td_rate'] = np.where(
            df['rush_attempts'] > 0,
            df['rush_tds'] / df['rush_attempts'],
            0
        )
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
        """Create rolling average features.
        
        Args:
            df: DataFrame with basic features
            groupby_col: Column to group by (team or opponent)
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        # Features to calculate rolling averages for
        rolling_cols = [
            'pass_attempts', 'pass_completions', 'pass_yards', 'pass_tds',
            'rush_attempts', 'rush_yards', 'rush_tds',
            'total_plays', 'total_yards', 'total_tds',
            'pass_rate', 'completion_rate', 'yards_per_pass', 'yards_per_rush',
            'pass_td_rate', 'rush_td_rate'
        ]
        
        for window in self.rolling_windows:
            for col in rolling_cols:
                feature_name = f'{groupby_col}_{col}_avg_{window}g'
                
                # Calculate rolling mean (excluding current game)
                df[feature_name] = df.groupby(groupby_col)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
        
        return df
    
    def _create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on opponent's defensive performance.
        
        Args:
            df: DataFrame with team features
            
        Returns:
            DataFrame with opponent features
        """
        logger.info("Creating opponent features")
        
        # Create defensive stats (what opponents allowed)
        defense_stats = df.groupby(['opponent', 'season', 'week']).agg({
            'pass_yards': 'sum',
            'rush_yards': 'sum',
            'pass_tds': 'sum',
            'rush_tds': 'sum',
            'total_yards': 'sum',
            'total_tds': 'sum'
        }).reset_index()
        
        defense_stats.columns = ['team', 'season', 'week'] + \
                               [f'def_{col}_allowed' for col in defense_stats.columns[3:]]
        
        # Sort and create rolling features for defense
        defense_stats = defense_stats.sort_values(['team', 'season', 'week'])
        
        for window in self.rolling_windows:
            for col in [c for c in defense_stats.columns if 'allowed' in c]:
                feature_name = f'{col}_avg_{window}g'
                defense_stats[feature_name] = defense_stats.groupby('team')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
        
        # Merge opponent defensive features
        opponent_features = defense_stats.drop(
            columns=[c for c in defense_stats.columns if 'allowed' in c and 'avg' not in c]
        )
        
        df = df.merge(
            opponent_features,
            left_on=['opponent', 'season', 'week'],
            right_on=['team', 'season', 'week'],
            suffixes=('', '_opp')
        )
        
        # Drop duplicate columns
        df = df.drop(columns=['team_opp'])
        
        return df
    
    def _create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team efficiency features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with efficiency features
        """
        # Offensive efficiency trends
        for window in self.rolling_windows:
            # Yards per play trend
            df[f'yards_per_play_trend_{window}g'] = (
                df[f'team_total_yards_avg_{window}g'] / 
                df[f'team_total_plays_avg_{window}g']
            ).fillna(0)
            
            # TD efficiency trend
            df[f'td_per_yard_trend_{window}g'] = np.where(
                df[f'team_total_yards_avg_{window}g'] > 0,
                df[f'team_total_tds_avg_{window}g'] / df[f'team_total_yards_avg_{window}g'],
                0
            )
        
        return df
    
    def _create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create game context features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with context features
        """
        # Week of season (early, mid, late)
        df['early_season'] = (df['week'] <= 6).astype(int)
        df['mid_season'] = ((df['week'] > 6) & (df['week'] <= 12)).astype(int)
        df['late_season'] = (df['week'] > 12).astype(int)
        
        # Division game (simplified - would need division data for accuracy)
        # For now, just a placeholder
        df['division_game'] = 0
        
        # Rest days (simplified - assumes normal Sunday-to-Sunday)
        df['rest_days'] = 7
        
        return df
    
    def prepare_prediction_features(
        self,
        historical_data: pd.DataFrame,
        team: str,
        opponent: str,
        week: int,
        season: int,
        home: bool
    ) -> pd.DataFrame:
        """Prepare features for making a prediction.
        
        Args:
            historical_data: Historical team data with features
            team: Team abbreviation
            opponent: Opponent abbreviation
            week: Week number
            season: Season year
            home: Whether team is home
            
        Returns:
            DataFrame with one row of features for prediction
        """
        # Get most recent features for team
        team_features = historical_data[
            historical_data['team'] == team
        ].iloc[-1:].copy()
        
        # Get most recent defensive features for opponent
        opp_features = historical_data[
            historical_data['team'] == opponent
        ].iloc[-1:]
        
        # Update with current game info
        team_features['opponent'] = opponent
        team_features['week'] = week
        team_features['season'] = season
        team_features['home'] = int(home)
        
        # Update context features
        team_features['early_season'] = int(week <= 6)
        team_features['mid_season'] = int(6 < week <= 12)
        team_features['late_season'] = int(week > 12)
        
        # Get opponent defensive averages
        opp_def_cols = [c for c in opp_features.columns if 'def_' in c and 'avg' in c]
        for col in opp_def_cols:
            team_features[col] = opp_features[col].values[0]
        
        return team_features