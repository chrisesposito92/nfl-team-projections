from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from pathlib import Path
from .config import TEAM_TARGETS, ARTIFACTS_DIR, TEAM_MODEL_DIRNAME
from .utils import ensure_dir

class TeamProjectionModel:
    def __init__(self, projection_year: int, seed: int = 42):
        self.projection_year = projection_year
        self.seed = seed
        self.models: Dict[str, Pipeline] = {}
        self.feature_cols: List[str] = []
        self.model_dir = Path(ARTIFACTS_DIR) / TEAM_MODEL_DIRNAME / str(projection_year)
        ensure_dir(self.model_dir)

    def fit(self, df: pd.DataFrame):
        if (df["season"] >= self.projection_year).any():
            raise ValueError("Training data contains the projection year or beyond; this would cause leakage.")
        # Choose numeric rolling features + is_home
        num_cols = [c for c in df.columns if c.startswith("r") and c.endswith("_mean")]
        cols = num_cols + ["is_home"]
        self.feature_cols = cols
        X = df[cols].fillna(0.0).astype(float)
        for target in TEAM_TARGETS:
            y = df[target].astype(float).values
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ])
            pipe.fit(X, y)
            self.models[target] = pipe
        self._save()

    def predict(self, features_row: pd.DataFrame) -> Dict[str, float]:
        if not self.models:
            self._load()
        X = features_row[self.feature_cols].fillna(0.0).astype(float)
        out = {}
        for target, model in self.models.items():
            yhat = float(model.predict(X)[0])
            out[target] = max(0.0, yhat)
        return out

    def _save(self):
        for t, m in self.models.items():
            joblib.dump({"model": m, "feature_cols": self.feature_cols},
                        self.model_dir / f"{t}.joblib")

    def _load(self):
        if not self.models:
            any_file = list(self.model_dir.glob("*.joblib"))
            if not any_file:
                raise FileNotFoundError(f"No cached team models found under {self.model_dir}. Train first.")
            # Restore feature cols consistently from first file
            blob = joblib.load(any_file[0])
            self.feature_cols = blob["feature_cols"]
            for fp in any_file:
                t = fp.stem
                blob = joblib.load(fp)
                self.models[t] = blob["model"]
