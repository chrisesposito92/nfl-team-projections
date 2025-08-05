"""Unit tests for input validators."""

import pytest
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils.validators import InputValidator


class TestInputValidator:
    """Test input validation."""
    
    def setup_method(self):
        """Set up test validator."""
        self.validator = InputValidator()
    
    def test_validate_year_valid(self):
        """Test valid year inputs."""
        current_year = datetime.now().year
        
        # Current year should be valid
        assert self.validator.validate_year(str(current_year))
        
        # Future years (up to 5) should be valid
        assert self.validator.validate_year(str(current_year + 1))
        assert self.validator.validate_year(str(current_year + 5))
        
        # Recent historical years should be valid
        assert self.validator.validate_year(str(current_year - 1))
        assert self.validator.validate_year(str(current_year - 5))
    
    def test_validate_year_invalid(self):
        """Test invalid year inputs."""
        current_year = datetime.now().year
        
        # Too far in future
        assert not self.validator.validate_year(str(current_year + 10))
        
        # Too far in past
        assert not self.validator.validate_year(str(current_year - 20))
        
        # Non-numeric
        assert not self.validator.validate_year("abc")
        assert not self.validator.validate_year("")
    
    def test_validate_team_valid(self):
        """Test valid team inputs."""
        valid_teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR']
        
        assert self.validator.validate_team('ARI', valid_teams)
        assert self.validator.validate_team('ari', valid_teams)  # Case insensitive
        assert self.validator.validate_team('BUF', valid_teams)
    
    def test_validate_team_invalid(self):
        """Test invalid team inputs."""
        valid_teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR']
        
        assert not self.validator.validate_team('XXX', valid_teams)
        assert not self.validator.validate_team('', valid_teams)
        assert not self.validator.validate_team('ARIZONA', valid_teams)
    
    def test_validate_week_valid(self):
        """Test valid week inputs."""
        valid_weeks = [1, 2, 3, 4, 5]
        
        assert self.validator.validate_week('1', valid_weeks)
        assert self.validator.validate_week('3', valid_weeks)
        assert self.validator.validate_week('5', valid_weeks)
    
    def test_validate_week_invalid(self):
        """Test invalid week inputs."""
        valid_weeks = [1, 2, 3, 4, 5]
        
        assert not self.validator.validate_week('0', valid_weeks)
        assert not self.validator.validate_week('6', valid_weeks)
        assert not self.validator.validate_week('abc', valid_weeks)
        assert not self.validator.validate_week('', valid_weeks)