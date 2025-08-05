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
    
    def create_features(self, player_stats: pd.DataFrame, seasonal_data: pd.DataFrame = None,
                       depth_charts: pd.DataFrame = None) -> pd.DataFrame:
        """Create all features for player share projections.
        
        Args:
            player_stats: DataFrame with player-game statistics
            seasonal_data: Optional historical seasonal data with target shares
            depth_charts: Optional depth chart data
            
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
        
        # Add historical seasonal shares if available
        if seasonal_data is not None:
            features = self._add_historical_shares(features, seasonal_data)
        
        # Add depth chart features if available
        if depth_charts is not None:
            features = self._add_depth_features(features, depth_charts)
        
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
        season: int,
        seasonal_data: pd.DataFrame = None
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
        logger.info(f"Preparing prediction features for {len(active_players)} active players")
        
        if active_players.empty:
            logger.warning(f"No active players found for {team} in week {week} of {season}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame()
        
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
                    'player_display_name': player.get('player_display_name', player.get('player_name', 'Unknown')),
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
                
                # Add historical shares from seasonal data if available
                if seasonal_data is not None and player_id in seasonal_data['player_id'].values:
                    player_seasonal = seasonal_data[seasonal_data['player_id'] == player_id]
                    if not player_seasonal.empty:
                        # Get most recent season
                        latest_season = player_seasonal.iloc[-1]
                        for col in ['tgt_sh', 'ay_sh', 'ry_sh', 'wopr_y', 'dom']:
                            if col in latest_season:
                                # Rename wopr_y to just wopr in the features
                                feature_name = col if col != 'wopr_y' else 'wopr'
                                player_features[f'last_season_{feature_name}'] = latest_season[col]
            
            # Update current game info
            player_features['season'] = season
            player_features['week'] = week
            player_features['recent_team'] = team
            
            prediction_features.append(player_features)
        
        if not prediction_features:
            logger.warning("No prediction features could be created for any player")
            return pd.DataFrame()
            
        return pd.concat(prediction_features, ignore_index=True)
    
    def _add_historical_shares(self, features: pd.DataFrame, seasonal_data: pd.DataFrame) -> pd.DataFrame:
        """Add historical target share data from seasonal stats.
        
        Args:
            features: Current feature DataFrame
            seasonal_data: Historical seasonal data with shares
            
        Returns:
            Features with historical shares added
        """
        logger.info("Adding historical share features")
        
        # Key seasonal share columns from the data
        # Note: wopr_y is the custom calculated WOPR, wopr_x is from nfl_data_py
        share_cols = ['tgt_sh', 'ay_sh', 'ry_sh', 'rtd_sh', 'wopr_y', 'dom']
        
        # Get most recent season for each player
        most_recent = seasonal_data.sort_values(['player_id', 'season']).groupby('player_id').last()
        
        # Add last season shares
        for col in share_cols:
            if col in most_recent.columns:
                # Rename wopr_y to just wopr in the features
                feature_name = col if col != 'wopr_y' else 'wopr'
                features[f'last_season_{feature_name}'] = features['player_id'].map(
                    most_recent[col].to_dict()
                ).fillna(0)
        
        # Add career averages
        # Filter to only existing columns
        existing_share_cols = [col for col in share_cols if col in seasonal_data.columns]
        career_avg = seasonal_data.groupby('player_id')[existing_share_cols].mean()
        
        for col in existing_share_cols:
            if col in career_avg.columns:
                feature_name = col if col != 'wopr_y' else 'wopr'
                features[f'career_avg_{feature_name}'] = features['player_id'].map(
                    career_avg[col].to_dict()
                ).fillna(0)
        
        # Add last 3 year average (more recent trend)
        current_year = features['season'].max()
        recent_years = seasonal_data[seasonal_data['season'] >= current_year - 3]
        recent_avg = recent_years.groupby('player_id')[existing_share_cols].mean()
        
        for col in existing_share_cols:
            if col in recent_avg.columns:
                feature_name = col if col != 'wopr_y' else 'wopr'
                features[f'recent_avg_{feature_name}'] = features['player_id'].map(
                    recent_avg[col].to_dict()
                ).fillna(0)
        
        return features
    
    def _add_depth_features(self, features: pd.DataFrame, depth_charts: pd.DataFrame) -> pd.DataFrame:
        """Add depth chart-based features.
        
        Args:
            features: Current feature DataFrame
            depth_charts: Depth chart data
            
        Returns:
            Features with depth chart features added
        """
        logger.info("Adding depth chart features")
        
        # Get most recent depth chart for each player
        if 'week' in depth_charts.columns and 'gsis_id' in depth_charts.columns:
            # Group by gsis_id (player_id) and get most recent
            depth_features = depth_charts.sort_values(['gsis_id', 'season', 'week']).groupby('gsis_id').last()
        else:
            # Use player_id if available
            depth_features = depth_charts.groupby('player_id').last()
        
        # Add depth rank (1 = starter, 2 = backup, etc.)
        if 'depth_team' in depth_features.columns:
            features['depth_rank'] = features['player_id'].map(
                depth_features['depth_team'].to_dict()
            ).fillna(4)  # Assume 4th string if not on depth chart
        
        # Binary starter flag
        features['is_starter'] = (features.get('depth_rank', 4) == 1).astype(int)
        
        # Add position-specific depth rank (e.g., WR1, WR2, WR3)
        features['position_depth_rank'] = features.groupby(
            ['recent_team', 'position']
        )['target_share'].rank(method='dense', ascending=False)
        
        # Is top 3 at position
        features['is_top3_position'] = (features['position_depth_rank'] <= 3).astype(int)
        
        return features