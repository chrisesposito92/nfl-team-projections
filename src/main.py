"""Main CLI interface for NFL Projections."""

import click
import pandas as pd
import logging
from typing import Optional
from pathlib import Path
import sys

from data.loader import NFLDataLoader
from data.aggregator import DataAggregator
from features.team_features import TeamFeatureEngineer
from features.player_features import PlayerFeatureEngineer
from models.team_model import TeamProjectionModel
from models.player_model import PlayerShareModel
from projections.calculator import ProjectionCalculator
from utils.validators import InputValidator
from utils.exceptions import DataLoadError, InsufficientDataError, ModelNotTrainedError
from config import MIN_TRAINING_YEARS, SEASON_WEEKS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLProjectionsCLI:
    """Main CLI application for NFL projections."""
    
    def __init__(self):
        self.loader = NFLDataLoader()
        self.aggregator = DataAggregator()
        self.team_feature_engineer = TeamFeatureEngineer()
        self.player_feature_engineer = PlayerFeatureEngineer()
        self.team_model = TeamProjectionModel()
        self.player_model = PlayerShareModel()
        self.calculator = ProjectionCalculator()
        self.validator = InputValidator()
        
    def run(self):
        """Run the interactive CLI."""
        click.clear()
        click.secho("NFL Offensive Projections Modeler V1.0", bold=True, fg='green')
        click.echo("=" * 50)
        
        try:
            # Get year
            year = self._prompt_year()
            
            # Get team
            team = self._prompt_team()
            
            # Get timeframe
            full_season, week = self._prompt_timeframe(year, team)
            
            # Check if models exist, train if needed
            self._ensure_models_trained(year)
            
            # Generate projections
            click.echo("\nGenerating projections...")
            
            if full_season:
                projections = self._project_full_season(year, team)
            else:
                projections = self._project_single_week(year, team, week)
            
            # Display results
            self._display_results(projections, team, year, week if not full_season else None)
            
        except KeyboardInterrupt:
            click.echo("\n\nExiting...")
            sys.exit(0)
        except Exception as e:
            click.secho(f"\nError: {str(e)}", fg='red')
            logger.exception("Unexpected error")
            sys.exit(1)
    
    def _prompt_year(self) -> int:
        """Prompt user for projection year."""
        while True:
            year_str = click.prompt(
                "\nPlease enter the year you would like to project (e.g., 2025)",
                type=str
            )
            
            if self.validator.validate_year(year_str):
                return int(year_str)
            else:
                click.secho("Invalid year. Please enter a valid year (e.g., 2025)", fg='red')
    
    def _prompt_team(self) -> str:
        """Prompt user for team selection."""
        # Load team descriptions
        teams_df = self.loader.load_team_descriptions()
        teams_df = teams_df[teams_df['team_abbr'].notna()].sort_values('team_abbr')
        
        click.echo("\nAvailable teams:")
        for _, team in teams_df.iterrows():
            click.echo(f"  {team['team_abbr']} - {team['team_name']}")
        
        while True:
            team = click.prompt(
                "\nPlease select a team from the list above (e.g., 'CIN')",
                type=str
            ).upper()
            
            if self.validator.validate_team(team, teams_df['team_abbr'].tolist()):
                return team
            else:
                click.secho("Invalid team. Please select from the list above.", fg='red')
    
    def _prompt_timeframe(self, year: int, team: str) -> tuple:
        """Prompt user for projection timeframe."""
        full_season = click.confirm("\nProject full REGULAR season?", default=True)
        
        if full_season:
            return True, None
        else:
            # Load schedule
            schedule = self.loader.get_future_schedule(year, team)
            
            if schedule.empty:
                click.secho(f"No schedule found for {team} in {year}", fg='red')
                sys.exit(1)
            
            # Display weekly matchups
            click.echo("\nWeekly matchups:")
            for idx, game in schedule.iterrows():
                if game['home_team'] == team:
                    opponent = game['away_team']
                    location = "vs"
                else:
                    opponent = game['home_team']
                    location = "@"
                
                click.echo(f"  {game['week']}. Week {game['week']} {location} {opponent}")
            
            while True:
                week_str = click.prompt(
                    "\nPlease select a week to project by its number",
                    type=str
                )
                
                if self.validator.validate_week(week_str, schedule['week'].tolist()):
                    return False, int(week_str)
                else:
                    click.secho("Invalid week. Please select from the list above.", fg='red')
    
    def _ensure_models_trained(self, target_year: int):
        """Ensure models are trained for the target year."""
        # Check if models exist
        try:
            self.team_model.load_models()
            self.player_model.load_models()
            logger.info("Loaded existing models")
        except:
            logger.info("Models not found, training new models...")
            click.echo("\nNo trained models found. Training models...")
            self._train_models(target_year)
    
    def _train_models(self, target_year: int):
        """Train all models."""
        # Load historical data
        training_years = list(range(target_year - MIN_TRAINING_YEARS - 1, target_year))
        
        if len(training_years) < MIN_TRAINING_YEARS:
            raise InsufficientDataError(
                f"Need at least {MIN_TRAINING_YEARS} years of data to train models"
            )
        
        click.echo(f"Loading data for years: {training_years}")
        
        try:
            with click.progressbar(length=100, label='Loading data') as bar:
                data = self.loader.load_all_data(training_years)
                bar.update(50)
                
                # Aggregate data
                team_stats = self.aggregator.aggregate_team_game_stats(data['pbp'])
                player_stats = self.aggregator.aggregate_player_game_stats(data['weekly'])
                player_stats = self.aggregator.merge_snap_data(player_stats, data['snaps'])
                bar.update(50)
        except Exception as e:
            raise DataLoadError(f"Failed to load data: {str(e)}")
        
        # Create training datasets
        datasets = self.aggregator.create_training_datasets(
            team_stats, player_stats, target_year
        )
        
        # Engineer features
        click.echo("Engineering features...")
        team_features = self.team_feature_engineer.create_features(datasets['team_train'])
        player_features = self.player_feature_engineer.create_features(datasets['player_train'])
        
        # Train models
        click.echo("Training team model...")
        team_metrics = self.team_model.train(team_features)
        
        click.echo("Training player model...")
        player_metrics = self.player_model.train(player_features)
        
        click.secho("Models trained successfully!", fg='green')
    
    def _project_single_week(self, year: int, team: str, week: int) -> pd.DataFrame:
        """Generate projections for a single week."""
        # Get schedule info
        schedule = self.loader.get_future_schedule(year, team)
        game = schedule[schedule['week'] == week].iloc[0]
        
        home = game['home_team'] == team
        opponent = game['away_team'] if home else game['home_team']
        
        # Load recent data for features
        # For now, use 2023 data since there seem to be issues with 2024
        recent_years = [year - 1, year - 2]
        
        # Download data if not cached
        click.echo(f"Loading data for years: {recent_years}")
        click.echo("This may take a few minutes if data needs to be downloaded...")
        
        try:
            recent_data = self.loader.load_all_data(recent_years)
        except Exception as e:
            click.secho(f"Error loading data: {e}", fg='red')
            click.echo("Trying with just 2023 data...")
            recent_data = self.loader.load_all_data([2023])
        
        # Prepare team features
        team_stats = self.aggregator.aggregate_team_game_stats(recent_data['pbp'])
        team_features = self.team_feature_engineer.create_features(team_stats)
        
        prediction_features = self.team_feature_engineer.prepare_prediction_features(
            team_features, team, opponent, week, year, home
        )
        
        # Generate team projections
        team_projections = self.team_model.predict(prediction_features)
        
        # Get active players
        active_players = self.loader.get_active_players(year, week, team)
        
        # Prepare player features
        player_stats = self.aggregator.aggregate_player_game_stats(recent_data['weekly'])
        player_stats = self.aggregator.merge_snap_data(player_stats, recent_data['snaps'])
        player_features = self.player_feature_engineer.create_features(player_stats)
        
        player_prediction_features = self.player_feature_engineer.prepare_prediction_features(
            player_features, active_players, team, week, year
        )
        
        # Generate player share projections
        player_shares = self.player_model.predict(player_prediction_features)
        player_shares = self.player_model.normalize_team_shares(player_shares, team)
        
        # Get player efficiency metrics
        player_efficiency = player_stats[
            player_stats['player_id'].isin(active_players['player_id'])
        ].groupby('player_id').last()
        
        # Calculate final projections
        projections = self.calculator.calculate_player_projections(
            team_projections, player_shares, player_efficiency
        )
        
        return projections
    
    def _project_full_season(self, year: int, team: str) -> pd.DataFrame:
        """Generate projections for full season."""
        weekly_projections = []
        
        with click.progressbar(
            range(1, SEASON_WEEKS + 1),
            label='Projecting weeks'
        ) as weeks:
            for week in weeks:
                try:
                    week_proj = self._project_single_week(year, team, week)
                    weekly_projections.append(week_proj)
                except Exception as e:
                    logger.warning(f"Failed to project week {week}: {e}")
        
        # Aggregate to season totals
        season_projections = self.calculator.aggregate_season_projections(weekly_projections)
        
        return season_projections
    
    def _display_results(self, projections: pd.DataFrame, team: str, year: int, week: Optional[int]):
        """Display projection results."""
        click.echo("\n" + "=" * 80)
        
        if week:
            click.secho(f"{year} Week {week} Projections - {team}", bold=True, fg='cyan')
        else:
            click.secho(f"{year} Season Projections - {team}", bold=True, fg='cyan')
        
        click.echo("=" * 80)
        
        # Format and display
        formatted = self.calculator.format_output(projections, week)
        
        # Convert to string for better display
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.1f}'.format)
        
        click.echo(formatted.to_string(index=False))
        
        click.echo("\n" + "=" * 80)


def main():
    """Main entry point."""
    cli = NFLProjectionsCLI()
    cli.run()


if __name__ == "__main__":
    main()