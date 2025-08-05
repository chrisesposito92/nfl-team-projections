"""Feature engineering for player-level projections."""

import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class PlayerFeatureEngineer:
    """Creates features for player share projection models."""
    
    def __init__(self, rolling_windows: List[int] = [3, 6, 12]):
        """Initialize feature engineer.
        
        Args:
            rolling_windows: List of game windows for rolling statistics
        """
        self.rolling_windows = rolling_windows
    
    def create_features(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Create all features for player share projections.
        
        Args:
            player_stats: DataFrame with player-game statistics
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating player features")
        
        # Sort by player and date
        player_stats = player_stats.sort_values(
            ['player_id', 'season', 'week']
        )
        
        # Create player context features
        features = self._create_player_context(player_stats)
        
        # Create usage trend features
        features = self._create_usage_trends(features)
        
        # Create team context features
        features = self._create_team_context(features)
        
        # Create position group features
        features = self._create_position_group_features(features)
        
        logger.info(f"Created {len(features.columns)} player features")
        
        return features
    
    def _create_player_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create player-specific context features.
        
        Args:
            df: Player statistics DataFrame
            
        Returns:
            DataFrame with player context features
        """
        df = df.copy()
        
        # Calculate player age (if birthdate available)
        # For now, use years in league as proxy
        df['seasons_played'] = df.groupby('player_id')['season'].transform(
            lambda x: x.rank(method='dense')
        )
        
        # Games played in current season
        df['games_played_season'] = df.groupby(['player_id', 'season']).cumcount()
        
        # Is starter (based on snap percentage)
        df['is_starter'] = (df['offense_pct'] > 0.5).astype(int)
        
        # Recent injury (was on injury report in last 3 games)
        # This would need injury data integration
        df['recent_injury'] = 0  # Placeholder
        
        return df
    
    def _create_usage_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on usage trends.
        
        Args:
            df: Player statistics DataFrame
            
        Returns:
            DataFrame with usage trend features
        """
        df = df.copy()
        
        # Features to track trends for
        usage_cols = [
            'targets', 'target_share', 'carries', 'rush_attempt_share',
            'offense_snaps', 'offense_pct', 'yards_per_target', 
            'yards_per_carry', 'catch_rate'
        ]
        
        for window in self.rolling_windows:
            for col in usage_cols:
                if col in df.columns:
                    # Rolling average
                    feature_name = f'{col}_avg_{window}g'
                    df[feature_name] = df.groupby('player_id')[col].transform(
                        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                    )
                    
                    # Rolling trend (current vs average)
                    if window == min(self.rolling_windows):
                        trend_name = f'{col}_trend'
                        df[trend_name] = df[col] - df[feature_name]
        
        # Career averages
        for col in usage_cols:
            if col in df.columns:
                df[f'{col}_career_avg'] = df.groupby('player_id')[col].transform(
                    lambda x: x.expanding().mean().shift(1)
                )
        
        return df
    
    def _create_team_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on team context.
        
        Args:
            df: Player statistics DataFrame
            
        Returns:
            DataFrame with team context features
        """
        df = df.copy()
        
        # Team passing volume trends
        team_pass_volume = df.groupby(['recent_team', 'season', 'week'])['targets'].sum()
        team_rush_volume = df.groupby(['recent_team', 'season', 'week'])['carries'].sum()
        
        # Merge back team volumes
        df = df.merge(
            team_pass_volume.rename('team_total_targets').reset_index(),
            on=['recent_team', 'season', 'week'],
            how='left'
        )
        
        df = df.merge(
            team_rush_volume.rename('team_total_carries').reset_index(),
            on=['recent_team', 'season', 'week'],
            how='left'
        )
        
        # Team volume trends
        for window in self.rolling_windows:
            df[f'team_targets_avg_{window}g'] = df.groupby('recent_team')['team_total_targets'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            
            df[f'team_carries_avg_{window}g'] = df.groupby('recent_team')['team_total_carries'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
        
        return df
    
    def _create_position_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on position group dynamics.
        
        Args:
            df: Player statistics DataFrame
            
        Returns:
            DataFrame with position group features
        """
        df = df.copy()
        
        # Count active players at same position
        position_counts = df.groupby(
            ['recent_team', 'season', 'week', 'position']
        )['player_id'].count().rename('position_player_count')
        
        df = df.merge(
            position_counts.reset_index(),
            on=['recent_team', 'season', 'week', 'position'],
            how='left'
        )
        
        # Player's rank within position by usage
        df['position_target_rank'] = df.groupby(
            ['recent_team', 'season', 'week', 'position']
        )['targets'].rank(ascending=False, method='min')
        
        df['position_carry_rank'] = df.groupby(
            ['recent_team', 'season', 'week', 'position']
        )['carries'].rank(ascending=False, method='min')
        
        # Share of position group's usage
        position_targets = df.groupby(
            ['recent_team', 'season', 'week', 'position']
        )['targets'].transform('sum')
        
        df['position_target_share'] = np.where(
            position_targets > 0,
            df['targets'] / position_targets,
            0
        )
        
        # Position-specific features
        df['is_qb'] = (df['position'] == 'QB').astype(int)
        df['is_rb'] = (df['position'] == 'RB').astype(int)
        df['is_wr'] = (df['position'] == 'WR').astype(int)
        df['is_te'] = (df['position'] == 'TE').astype(int)
        
        return df
    
    def prepare_prediction_features(
        self,
        historical_data: pd.DataFrame,
        active_players: pd.DataFrame,
        team: str,
        week: int,
        season: int
    ) -> pd.DataFrame:
        """Prepare features for active players for prediction.
        
        Args:
            historical_data: Historical player data with features
            active_players: DataFrame of active players for the game
            team: Team abbreviation
            week: Week number
            season: Season year
            
        Returns:
            DataFrame with features for all active players
        """
        prediction_features = []
        
        for _, player in active_players.iterrows():
            player_id = player['player_id']
            
            # Get player's most recent features
            player_history = historical_data[
                historical_data['player_id'] == player_id
            ]
            
            if len(player_history) > 0:
                # Use most recent data
                player_features = player_history.iloc[-1:].copy()
            else:
                # New player (likely rookie) - create baseline features
                player_features = pd.DataFrame([{
                    'player_id': player_id,
                    'player_display_name': player.get('player_display_name', 'Unknown'),
                    'position': player.get('position', 'Unknown'),
                    'recent_team': team,
                    'seasons_played': 0,
                    'games_played_season': 0,
                    'is_starter': 0,
                    'recent_injury': 0,
                    'offense_snaps': 0,
                    'offense_pct': 0.0,
                    'position_player_count': 5,  # Assume average position group size
                    'position_target_rank': 5,
                    'position_carry_rank': 5,
                    'position_target_share': 0.1,
                    'is_qb': int(player.get('position') == 'QB'),
                    'is_rb': int(player.get('position') == 'RB'),
                    'is_wr': int(player.get('position') == 'WR'),
                    'is_te': int(player.get('position') == 'TE'),
                }])
                
                # Add zero values for all rolling features
                for col in historical_data.columns:
                    if ('avg' in col or 'trend' in col) and col not in player_features.columns:
                        player_features[col] = 0
            
            # Update current game info
            player_features['season'] = season
            player_features['week'] = week
            player_features['recent_team'] = team
            
            prediction_features.append(player_features)
        
        return pd.concat(prediction_features, ignore_index=True)