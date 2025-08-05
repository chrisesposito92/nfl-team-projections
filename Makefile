.PHONY: install test clean run lint

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	pytest tests/

# Run unit tests only
test-unit:
	pytest tests/unit/ -v

# Run with coverage
test-coverage:
	pytest tests/ --cov=src --cov-report=html

# Clean up
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf models/
	rm -rf data/

# Run the CLI
run:
	python src/main.py

# Run linting (if installed)
lint:
	@echo "Install flake8 or ruff for linting"

# Create virtual environment with Python 3.12
venv:
	python3.12 -m venv venv
	@echo "Activate with: source venv/bin/activate"
	@echo "Then verify with: python --version"

# Download sample data (for testing)
download-data:
	python -c "import nfl_data_py as nfl; nfl.import_pbp_data([2023])"