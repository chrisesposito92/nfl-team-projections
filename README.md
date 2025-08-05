# NFL Offensive Projections Modeler V1.0

A machine learning system for forecasting NFL game and player statistics using hierarchical modeling.

## Overview

This project creates team-level offensive projections on a game-by-game basis and distributes these totals among individual offensive skill players (QB, RB, WR, TE).

## Requirements

- **Python 3.12** (required for nfl_data_py compatibility)
- pip
- virtualenv or venv

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd nfl-team-projections

# Create virtual environment with Python 3.12
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.12.x

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

Run the CLI application:

```bash
python src/main.py
```

Follow the prompts to:
1. Select a year to project
2. Choose an NFL team
3. Select full season or single week projection
4. View player projections

## Project Structure

```
nfl-team-projections/
├── src/
│   ├── data/          # Data ingestion modules
│   ├── features/      # Feature engineering
│   ├── models/        # ML models
│   └── utils/         # Utility functions
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── requirements.txt
└── setup.py
```

## Testing

Run tests with:

```bash
pytest tests/
```

## Data Source

All data is sourced from the [nfl_data_py](https://github.com/cooperdff/nfl_data_py) library.