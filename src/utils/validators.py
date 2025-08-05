"""Input validation utilities."""

import re
from typing import List
from datetime import datetime


class InputValidator:
    """Validates user inputs for the CLI."""
    
    def validate_year(self, year_str: str) -> bool:
        """Validate year input.
        
        Args:
            year_str: Year as string
            
        Returns:
            True if valid year
        """
        try:
            year = int(year_str)
            current_year = datetime.now().year
            
            # Allow projecting for current year and up to 5 years in future
            if current_year <= year <= current_year + 5:
                return True
            
            # Also allow recent historical years for testing
            if current_year - 10 <= year < current_year:
                return True
                
            return False
        except ValueError:
            return False
    
    def validate_team(self, team: str, valid_teams: List[str]) -> bool:
        """Validate team abbreviation.
        
        Args:
            team: Team abbreviation
            valid_teams: List of valid team abbreviations
            
        Returns:
            True if valid team
        """
        return team.upper() in valid_teams
    
    def validate_week(self, week_str: str, valid_weeks: List[int]) -> bool:
        """Validate week number.
        
        Args:
            week_str: Week as string
            valid_weeks: List of valid week numbers
            
        Returns:
            True if valid week
        """
        try:
            week = int(week_str)
            return week in valid_weeks
        except ValueError:
            return False