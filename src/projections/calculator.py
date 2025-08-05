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
        logger.info(f"Team projections columns: {team_projections.columns.tolist()}")
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
        
        # Apply position-based constraints first
        projections = self._apply_position_constraints(projections)
        
        # Normalize shares to ensure they sum to 1.0 for non-QB positions
        non_qb_mask = projections['position'] != 'QB'
        
        for col in ['proj_target_share', 'proj_pass_td_share']:
            if col in projections.columns:
                non_qb_total = projections.loc[non_qb_mask, col].sum()
                if non_qb_total > 0:
                    projections.loc[non_qb_mask, col] = projections.loc[non_qb_mask, col] / non_qb_total
                # Set QB shares to 0 for receiving stats
                projections.loc[~non_qb_mask, col] = 0
        
        # Normalize rush shares across all positions  
        for col in ['proj_rush_attempt_share', 'proj_rush_td_share']:
            if col in projections.columns:
                total = projections[col].sum()
                if total > 0:
                    projections[col] = projections[col] / total
        
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
        
        # Fill missing efficiency values with position-specific defaults
        projections.loc[projections['position'] == 'RB', 'catch_rate'] = projections.loc[projections['position'] == 'RB', 'catch_rate'].fillna(0.75)
        projections.loc[projections['position'] == 'WR', 'catch_rate'] = projections.loc[projections['position'] == 'WR', 'catch_rate'].fillna(0.65)
        projections.loc[projections['position'] == 'TE', 'catch_rate'] = projections.loc[projections['position'] == 'TE', 'catch_rate'].fillna(0.70)
        projections.loc[projections['position'] == 'QB', 'catch_rate'] = 0  # QBs don't catch passes
        
        projections.loc[projections['position'] == 'RB', 'yards_per_target'] = projections.loc[projections['position'] == 'RB', 'yards_per_target'].fillna(7.0)
        projections.loc[projections['position'] == 'WR', 'yards_per_target'] = projections.loc[projections['position'] == 'WR', 'yards_per_target'].fillna(10.0)
        projections.loc[projections['position'] == 'TE', 'yards_per_target'] = projections.loc[projections['position'] == 'TE', 'yards_per_target'].fillna(8.0)
        
        projections['yards_per_carry'] = projections['yards_per_carry'].fillna(4.0)
        
        # Calculate receiving stats (not for QBs)
        projections['proj_targets'] = (
            team_totals['pass_attempts'] * projections['proj_target_share']
        ).round()
        
        projections['proj_receptions'] = (
            projections['proj_targets'] * projections['catch_rate']
        ).round()
        
        # Calculate receiving yards proportionally to ensure they sum to team passing yards
        expected_yards = projections.loc[non_qb_mask, 'proj_targets'] * projections.loc[non_qb_mask, 'yards_per_target']
        total_expected = expected_yards.sum()
        
        projections['proj_receiving_yards'] = 0
        if total_expected > 0:
            projections.loc[non_qb_mask, 'proj_receiving_yards'] = (
                expected_yards * team_totals['pass_yards'] / total_expected
            ).round()
        
        # Calculate rushing stats
        projections['proj_rush_attempts'] = (
            team_totals['rush_attempts'] * projections['proj_rush_attempt_share']
        ).round()
        
        projections['proj_rushing_yards'] = (
            projections['proj_rush_attempts'] * projections['yards_per_carry']
        ).round()
        
        # Calculate touchdowns
        projections['proj_passing_tds'] = 0
        projections['proj_receiving_tds'] = 0
        projections['proj_rushing_tds'] = 0
        
        # Receiving TDs for non-QBs
        projections.loc[non_qb_mask, 'proj_receiving_tds'] = (
            team_totals['pass_tds'] * projections.loc[non_qb_mask, 'proj_pass_td_share']
        ).round()
        
        # Rushing TDs (all positions)
        projections['proj_rushing_tds'] = (
            team_totals['rush_tds'] * projections['proj_rush_td_share']
        ).round()
        
        # QB specific stats - only for starting QB
        projections['proj_pass_attempts'] = 0
        projections['proj_pass_completions'] = 0
        projections['proj_passing_yards'] = 0
        projections['proj_passing_tds'] = 0
        
        qb_mask = projections['position'] == 'QB'
        if qb_mask.any():
            qb_projections = projections[qb_mask]
            
            # Find starting QB (depth_team = 1 or first QB)
            starter_idx = None
            if 'depth_team' in qb_projections.columns:
                starter_mask = qb_projections['depth_team'] == 1
                if starter_mask.any():
                    starter_idx = qb_projections[starter_mask].index[0]
            
            if starter_idx is None:
                # No depth info or no depth_team=1, use first QB
                starter_idx = qb_projections.index[0]
                
            logger.info(f"Assigning passing stats to QB at index {starter_idx}")
            
            # Give all passing stats to starting QB
            projections.loc[starter_idx, 'proj_pass_attempts'] = team_totals['pass_attempts']
            projections.loc[starter_idx, 'proj_pass_completions'] = team_totals['pass_completions']
            projections.loc[starter_idx, 'proj_passing_yards'] = team_totals['pass_yards']
            projections.loc[starter_idx, 'proj_passing_tds'] = team_totals['pass_tds']
        
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
    
    def _apply_position_constraints(self, projections: pd.DataFrame) -> pd.DataFrame:
        """Apply realistic position-based constraints to shares.
        
        Args:
            projections: Raw share projections
            
        Returns:
            Constrained projections
        """
        # Sort by depth chart position if available
        if 'depth_team' in projections.columns:
            projections = projections.sort_values(['position', 'depth_team'])
        else:
            # Sort by predicted share to identify primary players
            projections = projections.sort_values(['position', 'proj_target_share'], ascending=[True, False])
        
        # Apply position-specific constraints
        for position in ['WR', 'RB', 'TE']:
            pos_mask = projections['position'] == position
            pos_players = projections[pos_mask]
            
            if len(pos_players) == 0:
                continue
                
            # Ensure realistic share distribution within position
            if position == 'WR':
                # WR1 typically gets 20-35% of targets
                # WR2 gets 15-25%
                # WR3 gets 10-20%
                for i, idx in enumerate(pos_players.index):
                    if i == 0:  # WR1
                        current_share = projections.loc[idx, 'proj_target_share']
                        projections.loc[idx, 'proj_target_share'] = max(0.20, min(0.35, current_share * 1.5))
                    elif i == 1:  # WR2
                        current_share = projections.loc[idx, 'proj_target_share']
                        projections.loc[idx, 'proj_target_share'] = max(0.15, min(0.25, current_share * 1.2))
                    elif i == 2:  # WR3
                        current_share = projections.loc[idx, 'proj_target_share']
                        projections.loc[idx, 'proj_target_share'] = max(0.10, min(0.20, current_share))
                    else:  # WR4+
                        projections.loc[idx, 'proj_target_share'] *= 0.5
                        
            elif position == 'RB':
                # RB1 typically gets 10-20% of targets
                # RB2 gets 5-15%
                for i, idx in enumerate(pos_players.index):
                    if i == 0:  # RB1
                        current_share = projections.loc[idx, 'proj_target_share']
                        projections.loc[idx, 'proj_target_share'] = max(0.10, min(0.20, current_share * 1.3))
                    elif i == 1:  # RB2
                        current_share = projections.loc[idx, 'proj_target_share']
                        projections.loc[idx, 'proj_target_share'] = max(0.05, min(0.15, current_share))
                    else:  # RB3+
                        projections.loc[idx, 'proj_target_share'] *= 0.3
                        
            elif position == 'TE':
                # TE1 typically gets 15-25% of targets
                # TE2 gets 5-10%
                for i, idx in enumerate(pos_players.index):
                    if i == 0:  # TE1
                        current_share = projections.loc[idx, 'proj_target_share']
                        projections.loc[idx, 'proj_target_share'] = max(0.15, min(0.25, current_share * 1.4))
                    else:  # TE2+
                        projections.loc[idx, 'proj_target_share'] *= 0.4
        
        # Apply rushing share constraints
        rb_mask = projections['position'] == 'RB'
        rb_players = projections[rb_mask].sort_values('proj_rush_attempt_share', ascending=False)
        
        for i, idx in enumerate(rb_players.index):
            if i == 0:  # RB1 gets 40-60% of RB rushes
                current_share = projections.loc[idx, 'proj_rush_attempt_share']
                projections.loc[idx, 'proj_rush_attempt_share'] = max(0.40, min(0.60, current_share * 1.5))
            elif i == 1:  # RB2 gets 20-40%
                current_share = projections.loc[idx, 'proj_rush_attempt_share']
                projections.loc[idx, 'proj_rush_attempt_share'] = max(0.20, min(0.40, current_share * 1.2))
            else:  # RB3+
                projections.loc[idx, 'proj_rush_attempt_share'] *= 0.5
        
        return projections
    
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
        
        # QBs shouldn't have receiving stats
        qb_mask = projections['position'] == 'QB'
        projections.loc[qb_mask, ['proj_targets', 'proj_receptions',
                                  'proj_receiving_yards', 'proj_receiving_tds']] = 0
        
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