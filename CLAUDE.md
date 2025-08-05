# NFL Team Projections - Project Context

## Project Overview
This is an NFL Offensive Projections Modeler that uses machine learning to forecast team and player statistics. It follows a hierarchical approach: first projecting team totals, then distributing those totals among players based on predicted usage shares.

## Important Requirements
- **Python 3.12 is REQUIRED** - The nfl_data_py library installation fails with other Python versions
- Always ensure virtual environments are created with Python 3.12

## Key Components

### Data Pipeline
- **Data Loader** (`src/data/loader.py`): Interfaces with nfl_data_py to load all required data
- **Data Aggregator** (`src/data/aggregator.py`): Transforms play-by-play data into model-ready format
- **Feature Engineers** (`src/features/`): Create rolling averages and contextual features

### Models
- **Team Model** (`src/models/team_model.py`): XGBoost models for 7 team-level stats
- **Player Model** (`src/models/player_model.py`): Predicts usage shares for players
- **Projection Calculator** (`src/projections/calculator.py`): Combines team/player predictions

### CLI Interface
- **Main** (`src/main.py`): Interactive CLI with guided prompts
- **Validators** (`src/utils/validators.py`): Input validation
- **Config** (`src/config.py`): Central configuration

## Testing Commands
- Run all tests: `pytest tests/`
- Run with coverage: `pytest tests/ --cov=src`
- Run specific test: `pytest tests/unit/test_validators.py -v`

## Common Tasks
- Install dependencies: `pip install -r requirements.txt`
- Run the CLI: `python src/main.py`
- Train new models: Delete the `models/` directory and run the CLI

## Known Limitations (V1.0)
- Rookies get minimal projections due to lack of historical data
- No playoff projections
- No weather or betting line integration
- Regular season only (17 games)

## Future Enhancements (Post V1.0)
- Integrate betting lines and weather data
- Add rookie projection model using combine/draft data
- Support playoff projections
- Web-based interface