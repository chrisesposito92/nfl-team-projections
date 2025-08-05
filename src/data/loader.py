"""Data loading module using nfl_data_py."""

import logging
from typing import List, Optional
import pandas as pd
import nfl_data_py as nfl
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NFLDataLoader:
    """Handles all data loading from nfl_data_py."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the data loader.
        
        Args:
            cache_dir: Optional directory for caching data
        """
        self.cache_dir = cache_dir
        
    def load_pbp_data(self, years: List[int]) -> pd.DataFrame:
        """Load play-by-play data for specified years.
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with play-by-play data
        """
        logger.info(f"Loading play-by-play data for years: {years}")
        
        # Try loading all years at once first (like your working example)
        try:
            logger.info("Attempting to load all years together...")
            logger.info(years)
            pbp = nfl.import_pbp_data([2024, 2023])
            return pbp
        except Exception as e:
            logger.warning(f"Failed to load all years together: {e}")
            
        # If that fails, try each year separately
        all_data = []
        for year in years:
            try:
                logger.info(f"Loading {year} play-by-play data individually...")
                year_data = nfl.import_pbp_data([year])
                all_data.append(year_data)
                logger.info(f"Successfully loaded {year} data")
            except Exception as e:
                logger.error(f"Failed to load {year} data: {type(e).__name__}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No play-by-play data could be loaded for any year")
            
        return pd.concat(all_data, ignore_index=True)
    
    def load_weekly_data(self, years: List[int]) -> pd.DataFrame:
        """Load weekly player data for specified years.
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with weekly player data
        """
        logger.info(f"Loading weekly data for years: {years}")
        return nfl.import_weekly_data(years)
    
    def load_snap_counts(self, years: List[int]) -> pd.DataFrame:
        """Load snap count data for specified years.
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with snap count data
        """
        logger.info(f"Loading snap counts for years: {years}")
        return nfl.import_snap_counts(years)
    
    def load_schedules(self, years: List[int]) -> pd.DataFrame:
        """Load schedule data for specified years.
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with schedule data
        """
        logger.info(f"Loading schedules for years: {years}")
        return nfl.import_schedules(years)
    
    def load_rosters(self, years: List[int]) -> pd.DataFrame:
        """Load roster data for specified years.
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with roster data
        """
        logger.info(f"Loading rosters for years: {years}")
        try:
            rosters = nfl.import_weekly_rosters(years)
            # Remove any duplicate indices
            if rosters.index.duplicated().any():
                logger.warning("Found duplicate indices in roster data, removing...")
                rosters = rosters[~rosters.index.duplicated(keep='first')]
            return rosters
        except Exception as e:
            logger.warning(f"Error loading rosters: {e}")
            # Try loading years separately
            all_rosters = []
            for year in years:
                try:
                    year_roster = nfl.import_weekly_rosters([year])
                    all_rosters.append(year_roster)
                except Exception as year_e:
                    logger.error(f"Could not load {year} roster data: {year_e}")
            
            if all_rosters:
                return pd.concat(all_rosters, ignore_index=True)
            else:
                raise ValueError("Could not load roster data for any year")
    
    def load_injuries(self, years: List[int]) -> pd.DataFrame:
        """Load injury data for specified years.
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with injury data
        """
        logger.info(f"Loading injury data for years: {years}")
        return nfl.import_injuries(years)
    
    def load_depth_charts(self, years: List[int]) -> pd.DataFrame:
        """Load depth chart data for specified years.
        
        Args:
            years: List of years to load
            
        Returns:
            DataFrame with depth chart data
        """
        logger.info(f"Loading depth charts for years: {years}")
        return nfl.import_depth_charts(years)
    
    def load_team_descriptions(self) -> pd.DataFrame:
        """Load team descriptions and abbreviations.
        
        Returns:
            DataFrame with team descriptions
        """
        logger.info("Loading team descriptions")
        return nfl.import_team_desc()
    
    def load_all_data(self, years: List[int]) -> dict:
        """Load all required data for the specified years.
        
        Args:
            years: List of years to load
            
        Returns:
            Dictionary with all loaded DataFrames
        """
        logger.info(f"Loading all data for years: {years}")
        
        data = {}
        
        with tqdm(total=7, desc="Loading NFL data") as pbar:
            data['pbp'] = self.load_pbp_data(years)
            pbar.update(1)
            
            data['weekly'] = self.load_weekly_data(years)
            pbar.update(1)
            
            data['snaps'] = self.load_snap_counts(years)
            pbar.update(1)
            
            data['schedules'] = self.load_schedules(years)
            pbar.update(1)
            
            data['rosters'] = self.load_rosters(years)
            pbar.update(1)
            
            data['injuries'] = self.load_injuries(years)
            pbar.update(1)
            
            data['teams'] = self.load_team_descriptions()
            pbar.update(1)
        
        logger.info("All data loaded successfully")
        return data
    
    def get_future_schedule(self, year: int, team: Optional[str] = None) -> pd.DataFrame:
        """Get future schedule for a specific year and optionally a team.
        
        Args:
            year: Year to get schedule for
            team: Optional team abbreviation to filter by
            
        Returns:
            DataFrame with future schedule
        """
        schedule = self.load_schedules([year])
        
        if team:
            schedule = schedule[
                (schedule['home_team'] == team) | 
                (schedule['away_team'] == team)
            ]
        
        return schedule
    
    def get_active_players(self, year: int, week: int, team: str) -> pd.DataFrame:
        """Get active players for a specific team and week using depth charts.
        
        Args:
            year: Year
            week: Week number
            team: Team abbreviation
            
        Returns:
            DataFrame with active players
        """
        # Try to load depth chart data first
        depth_year = year
        depth_charts = None
        
        # Try current year first, then work backwards
        for attempt_year in range(year, year - 3, -1):
            try:
                depth_charts = self.load_depth_charts([attempt_year])
                if not depth_charts.empty:
                    depth_year = attempt_year
                    logger.info(f"Using {depth_year} depth charts for {year} projections")
                    break
            except Exception as e:
                logger.debug(f"Could not load {attempt_year} depth charts: {e}")
                continue
        
        if depth_charts is not None and not depth_charts.empty:
            # Filter to team and relevant positions
            offensive_positions = ['QB', 'RB', 'WR', 'TE', 'FB']
            
            # Filter to team
            team_depth = depth_charts[depth_charts['team'] == team]
            
            # Filter to offensive positions
            offensive_pos_abbrevs = ['QB', 'RB', 'WR', 'TE', 'FB', 'HB', 'FL']
            if not team_depth.empty:
                team_depth = team_depth[
                    team_depth['pos_abb'].isin(offensive_pos_abbrevs) &
                    (team_depth['pos_rank'] <= 3)  # Top 3 on depth chart
                ]
                
                # Rename/map columns to match expected format
                team_depth['position'] = team_depth['pos_abb']
                team_depth['depth_team'] = team_depth['pos_rank']
                
                # Use gsis_id as player_id
                if 'gsis_id' in team_depth.columns:
                    team_depth['player_id'] = team_depth['gsis_id']
                else:
                    # Create a pseudo player_id from name
                    team_depth['player_id'] = team_depth['player_name'].str.replace(' ', '_').str.lower()
                
                logger.info(f"Found {len(team_depth)} offensive players from depth chart")
                return team_depth
        
        # Fallback to roster method if depth charts unavailable
        logger.info("Depth charts not available, falling back to roster data")
        
        # Original roster-based logic as fallback
        rosters = None
        roster_year = year
        
        for attempt_year in range(year, year - 5, -1):
            try:
                rosters = self.load_rosters([attempt_year])
                roster_year = attempt_year
                logger.info(f"Using {roster_year} roster data for {year} projections")
                break
            except Exception as e:
                logger.debug(f"Could not load {attempt_year} rosters: {e}")
                continue
        
        if rosters is None:
            raise ValueError(f"Could not load roster or depth chart data for {year}")
        
        # Filter to specific team and offensive positions
        offensive_positions = ['QB', 'RB', 'WR', 'TE', 'FB']
        team_roster = rosters[
            (rosters['team'] == team) & 
            (rosters['position'].isin(offensive_positions))
        ]
        
        logger.info(f"Found {len(team_roster)} offensive players from roster")
        
        return team_roster