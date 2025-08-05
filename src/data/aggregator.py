"""Data aggregation module for creating team and player level statistics."""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DataAggregator:
    """Aggregates play-by-play and weekly data into model-ready format."""
    
    def aggregate_team_game_stats(self, pbp_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate play-by-play data to team-game level.
        
        Args:
            pbp_data: Play-by-play data
            
        Returns:
            DataFrame with team-game level statistics
        """
        logger.info("Aggregating team-game statistics")
        
        # Filter to only regular season
        pbp_data = pbp_data[pbp_data['season_type'] == 'REG'].copy()
        
        # Filter to only offensive plays
        offensive_plays = pbp_data[
            (pbp_data['play_type'].isin(['pass', 'run'])) &
            (pbp_data['two_point_attempt'] == 0)
        ].copy()
        
        team_stats = []
        
        # Group by game and team
        for (game_id, posteam), plays in offensive_plays.groupby(['game_id', 'posteam']):
            if pd.isna(posteam):
                continue
                
            game_info = plays.iloc[0]
            
            stats = {
                'game_id': game_id,
                'season': game_info['season'],
                'week': game_info['week'],
                'team': posteam,
                'opponent': game_info['defteam'],
                'home': 1 if game_info['home_team'] == posteam else 0,
                
                # Passing stats
                'pass_attempts': len(plays[plays['play_type'] == 'pass']),
                'pass_completions': len(plays[
                    (plays['play_type'] == 'pass') & 
                    (plays['complete_pass'] == 1)
                ]),
                'pass_yards': plays[plays['play_type'] == 'pass']['yards_gained'].sum(),
                'pass_tds': len(plays[
                    (plays['play_type'] == 'pass') & 
                    (plays['touchdown'] == 1)
                ]),
                
                # Rushing stats
                'rush_attempts': len(plays[plays['play_type'] == 'run']),
                'rush_yards': plays[plays['play_type'] == 'run']['yards_gained'].sum(),
                'rush_tds': len(plays[
                    (plays['play_type'] == 'run') & 
                    (plays['touchdown'] == 1)
                ]),
                
                # Additional context
                'total_plays': len(plays),
                'total_yards': plays['yards_gained'].sum(),
                'total_tds': plays['touchdown'].sum()
            }
            
            team_stats.append(stats)
        
        team_df = pd.DataFrame(team_stats)
        logger.info(f"Aggregated {len(team_df)} team-game records")
        
        return team_df
    
    def aggregate_player_game_stats(self, weekly_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate weekly data to calculate player shares.
        
        Args:
            weekly_data: Weekly player data
            
        Returns:
            DataFrame with player-game statistics and shares
        """
        logger.info("Aggregating player-game statistics")
        
        # Filter to skill positions
        skill_players = weekly_data[
            weekly_data['position'].isin(['QB', 'RB', 'WR', 'TE'])
        ].copy()
        
        # Calculate team totals for share calculations
        team_totals = skill_players.groupby(['recent_team', 'season', 'week']).agg({
            'targets': 'sum',
            'carries': 'sum',
            'passing_tds': 'sum',
            'rushing_tds': 'sum',
            'receiving_tds': 'sum'
        }).reset_index()
        
        team_totals.columns = [
            'recent_team', 'season', 'week',
            'team_targets', 'team_carries', 
            'team_pass_tds', 'team_rush_tds', 'team_rec_tds'
        ]
        
        # Merge team totals back
        player_stats = skill_players.merge(
            team_totals,
            on=['recent_team', 'season', 'week'],
            how='left'
        )
        
        # Calculate shares (handle division by zero)
        player_stats['target_share'] = np.where(
            player_stats['team_targets'] > 0,
            player_stats['targets'] / player_stats['team_targets'],
            0
        )
        
        player_stats['rush_attempt_share'] = np.where(
            player_stats['team_carries'] > 0,
            player_stats['carries'] / player_stats['team_carries'],
            0
        )
        
        # Calculate TD shares by position
        player_stats['pass_td_share'] = 0
        player_stats['rush_td_share'] = 0
        
        # QB gets pass TD share
        qb_mask = player_stats['position'] == 'QB'
        player_stats.loc[qb_mask, 'pass_td_share'] = np.where(
            player_stats.loc[qb_mask, 'team_pass_tds'] > 0,
            player_stats.loc[qb_mask, 'passing_tds'] / player_stats.loc[qb_mask, 'team_pass_tds'],
            0
        )
        
        # RB/WR/TE get receiving TD share (from pass TDs)
        rec_mask = player_stats['position'].isin(['RB', 'WR', 'TE'])
        player_stats.loc[rec_mask, 'pass_td_share'] = np.where(
            player_stats.loc[rec_mask, 'team_rec_tds'] > 0,
            player_stats.loc[rec_mask, 'receiving_tds'] / player_stats.loc[rec_mask, 'team_rec_tds'],
            0
        ).astype(float)
        
        # All positions can get rush TD share
        player_stats['rush_td_share'] = np.where(
            player_stats['team_rush_tds'] > 0,
            player_stats['rushing_tds'] / player_stats['team_rush_tds'],
            0
        )
        
        # Add efficiency metrics
        player_stats['yards_per_target'] = np.where(
            player_stats['targets'] > 0,
            player_stats['receiving_yards'] / player_stats['targets'],
            0
        )
        
        player_stats['yards_per_carry'] = np.where(
            player_stats['carries'] > 0,
            player_stats['rushing_yards'] / player_stats['carries'],
            0
        )
        
        player_stats['catch_rate'] = np.where(
            player_stats['targets'] > 0,
            player_stats['receptions'] / player_stats['targets'],
            0
        )
        
        logger.info(f"Aggregated {len(player_stats)} player-game records")
        
        return player_stats
    
    def merge_snap_data(
        self, 
        player_stats: pd.DataFrame, 
        snap_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge snap count data with player statistics.
        
        Args:
            player_stats: Player statistics DataFrame
            snap_data: Snap count DataFrame
            
        Returns:
            Merged DataFrame with snap percentages
        """
        logger.info("Merging snap count data")
        
        # Standardize column names for merging
        snap_data = snap_data.rename(columns={
            'team': 'recent_team',
            'player': 'player_display_name'
        })
        
        # Merge on player, team, season, week
        merged = player_stats.merge(
            snap_data[['player_display_name', 'recent_team', 'season', 
                      'week', 'offense_snaps', 'offense_pct']],
            on=['player_display_name', 'recent_team', 'season', 'week'],
            how='left'
        )
        
        # Fill missing snap data with 0
        merged['offense_snaps'] = merged['offense_snaps'].fillna(0)
        merged['offense_pct'] = merged['offense_pct'].fillna(0)
        
        return merged
    
    def create_training_datasets(
        self,
        team_stats: pd.DataFrame,
        player_stats: pd.DataFrame,
        target_year: int
    ) -> Dict[str, pd.DataFrame]:
        """Create training datasets ensuring no future data leakage.
        
        Args:
            team_stats: Team-level statistics
            player_stats: Player-level statistics
            target_year: Year we want to make projections for
            
        Returns:
            Dictionary with 'team_train' and 'player_train' DataFrames
        """
        logger.info(f"Creating training datasets for target year {target_year}")
        
        # Filter to only include data before target year
        team_train = team_stats[team_stats['season'] < target_year].copy()
        player_train = player_stats[player_stats['season'] < target_year].copy()
        
        # Sort by date
        team_train = team_train.sort_values(['season', 'week'])
        player_train = player_train.sort_values(['season', 'week'])
        
        logger.info(f"Team training data: {len(team_train)} records from "
                   f"{team_train['season'].min()} to {team_train['season'].max()}")
        logger.info(f"Player training data: {len(player_train)} records from "
                   f"{player_train['season'].min()} to {player_train['season'].max()}")
        
        return {
            'team_train': team_train,
            'player_train': player_train
        }