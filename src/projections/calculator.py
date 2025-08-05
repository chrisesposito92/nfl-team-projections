"""Player stat calculation from team totals and shares."""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ProjectionCalculator:
    """Calculates final player projections from team totals and shares."""
    
    def calculate_player_projections(
        self,
        team_projections: pd.DataFrame,
        player_shares: pd.DataFrame,
        player_efficiency: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate final player stat projections.
        
        Args:
            team_projections: Team-level projections
            player_shares: Player share projections
            player_efficiency: Historical player efficiency metrics
            
        Returns:
            DataFrame with complete player projections
        """
        logger.info("Calculating player projections")
        logger.info(f"Team projections: {team_projections}")
        logger.info(f"Team totals: pass_attempts={team_projections['proj_pass_attempts'].iloc[0]:.1f}, pass_yards={team_projections['proj_pass_yards'].iloc[0]:.1f}")
        
        # Get team totals
        team_totals = {
            'pass_attempts': team_projections['proj_pass_attempts'].iloc[0],
            'pass_completions': team_projections['proj_pass_completions'].iloc[0],
            'pass_yards': team_projections['proj_pass_yards'].iloc[0],
            'pass_tds': team_projections['proj_pass_tds'].iloc[0],
            'rush_attempts': team_projections['proj_rush_attempts'].iloc[0],
            'rush_yards': team_projections['proj_rush_yards'].iloc[0],
            'rush_tds': team_projections['proj_rush_tds'].iloc[0]
        }
        
        projections = player_shares.copy()
        
        # Merge efficiency metrics
        if not player_efficiency.empty:
            # Reset index if player_id is the index
            if player_efficiency.index.name == 'player_id':
                player_efficiency = player_efficiency.reset_index()
            
            # Check which columns exist
            efficiency_cols = []
            for col in ['catch_rate', 'yards_per_target', 'yards_per_carry']:
                if col in player_efficiency.columns:
                    efficiency_cols.append(col)
            
            if efficiency_cols and 'player_id' in player_efficiency.columns:
                projections = projections.merge(
                    player_efficiency[['player_id'] + efficiency_cols],
                    on='player_id',
                    how='left'
                )
        
        # Fill missing efficiency values with defaults
        if 'catch_rate' not in projections.columns or projections['catch_rate'].isna().any():
            projections['catch_rate'] = projections.get('catch_rate', 0.65).fillna(0.65)
        if 'yards_per_target' not in projections.columns or projections['yards_per_target'].isna().any():
            projections['yards_per_target'] = projections.get('yards_per_target', 8.0).fillna(8.0)
        if 'yards_per_carry' not in projections.columns or projections['yards_per_carry'].isna().any():
            projections['yards_per_carry'] = projections.get('yards_per_carry', 4.0).fillna(4.0)
        
        # Calculate receiving stats
        projections['proj_targets'] = (
            team_totals['pass_attempts'] * projections['proj_target_share']
        ).round()
        
        projections['proj_receptions'] = (
            projections['proj_targets'] * projections['catch_rate']
        ).round()
        
        projections['proj_receiving_yards'] = (
            projections['proj_targets'] * projections['yards_per_target']
        ).round()
        
        # Calculate rushing stats
        projections['proj_rush_attempts'] = (
            team_totals['rush_attempts'] * projections['proj_rush_attempt_share']
        ).round()
        
        projections['proj_rushing_yards'] = (
            projections['proj_rush_attempts'] * projections['yards_per_carry']
        ).round()
        
        # Calculate touchdowns by position
        projections['proj_passing_tds'] = 0
        projections['proj_receiving_tds'] = 0
        projections['proj_rushing_tds'] = 0
        
        # QB passing TDs
        qb_mask = projections['position'] == 'QB'
        projections.loc[qb_mask, 'proj_passing_tds'] = (
            team_totals['pass_tds'] * projections.loc[qb_mask, 'proj_pass_td_share']
        ).round()
        
        # Skill position receiving TDs
        skill_mask = projections['position'].isin(['RB', 'WR', 'TE'])
        projections.loc[skill_mask, 'proj_receiving_tds'] = (
            team_totals['pass_tds'] * projections.loc[skill_mask, 'proj_pass_td_share']
        ).round()
        
        # Rushing TDs (all positions)
        projections['proj_rushing_tds'] = (
            team_totals['rush_tds'] * projections['proj_rush_td_share']
        ).round()
        
        # QB specific stats
        projections.loc[qb_mask, 'proj_pass_attempts'] = team_totals['pass_attempts']
        projections.loc[qb_mask, 'proj_pass_completions'] = team_totals['pass_completions']
        projections.loc[qb_mask, 'proj_passing_yards'] = team_totals['pass_yards']
        
        # Clean up projections
        projections = self._clean_projections(projections)
        
        # Select final columns
        final_columns = [
            'player_id', 'player_display_name', 'position', 'recent_team',
            'proj_pass_attempts', 'proj_pass_completions', 'proj_passing_yards', 
            'proj_passing_tds', 'proj_targets', 'proj_receptions', 
            'proj_receiving_yards', 'proj_receiving_tds', 'proj_rush_attempts',
            'proj_rushing_yards', 'proj_rushing_tds'
        ]
        
        return projections[final_columns]
    
    def _clean_projections(self, projections: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate projections.
        
        Args:
            projections: Raw projections
            
        Returns:
            Cleaned projections
        """
        # Ensure non-negative values
        numeric_cols = [col for col in projections.columns if 'proj_' in col]
        for col in numeric_cols:
            projections[col] = projections[col].clip(lower=0)
        
        # Position-specific cleaning
        # Non-QBs shouldn't have passing stats
        non_qb_mask = projections['position'] != 'QB'
        projections.loc[non_qb_mask, ['proj_pass_attempts', 'proj_pass_completions',
                                      'proj_passing_yards', 'proj_passing_tds']] = 0
        
        # Ensure receptions <= targets
        projections['proj_receptions'] = projections[['proj_receptions', 'proj_targets']].min(axis=1)
        
        # Fill NaN values with 0
        projections = projections.fillna(0)
        
        return projections
    
    def aggregate_season_projections(
        self,
        weekly_projections: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """Aggregate weekly projections to season totals.
        
        Args:
            weekly_projections: List of weekly projection DataFrames
            
        Returns:
            DataFrame with season total projections
        """
        # Concatenate all weeks
        all_weeks = pd.concat(weekly_projections, ignore_index=True)
        
        # Sum by player
        season_totals = all_weeks.groupby(['player_id', 'player_display_name', 
                                          'position', 'recent_team']).sum().reset_index()
        
        # Round to integers
        numeric_cols = [col for col in season_totals.columns if 'proj_' in col]
        season_totals[numeric_cols] = season_totals[numeric_cols].round()
        
        return season_totals
    
    def format_output(self, projections: pd.DataFrame, week: Optional[int] = None) -> pd.DataFrame:
        """Format projections for display.
        
        Args:
            projections: Player projections
            week: Week number (None for season totals)
            
        Returns:
            Formatted DataFrame for display
        """
        output = projections.copy()
        
        # Rename columns for display
        column_mapping = {
            'player_display_name': 'Player',
            'position': 'Pos',
            'recent_team': 'Team',
            'proj_pass_attempts': 'Pass Att',
            'proj_pass_completions': 'Pass Comp',
            'proj_passing_yards': 'Pass Yds',
            'proj_passing_tds': 'Pass TDs',
            'proj_targets': 'Targets',
            'proj_receptions': 'Rec',
            'proj_receiving_yards': 'Rec Yds',
            'proj_receiving_tds': 'Rec TDs',
            'proj_rush_attempts': 'Rush Att',
            'proj_rushing_yards': 'Rush Yds',
            'proj_rushing_tds': 'Rush TDs'
        }
        
        output = output.rename(columns=column_mapping)
        
        # Sort by position and projected points (simplified)
        output['proj_points'] = (
            output['Pass Yds'] * 0.04 +
            output['Pass TDs'] * 4 +
            output['Rec'] * 0.5 +
            output['Rec Yds'] * 0.1 +
            output['Rec TDs'] * 6 +
            output['Rush Yds'] * 0.1 +
            output['Rush TDs'] * 6
        )
        
        output = output.sort_values(['Pos', 'proj_points'], ascending=[True, False])
        
        # Drop player_id and proj_points for display
        output = output.drop(columns=['player_id', 'proj_points'])
        
        # Add week info if provided
        if week:
            output.insert(0, 'Week', week)
        
        return output