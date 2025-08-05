from setuptools import setup, find_packages

setup(
    name="nfl-team-projections",
    version="1.0.0",
    description="NFL Offensive Projections Modeler",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "nfl-data-py",
        "click>=8.1.0",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "nfl-projections=main:main",
        ],
    },
)