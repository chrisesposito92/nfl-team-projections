from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
from .config import ARTIFACTS_DIR, SHARE_MODEL_DIRNAME, SHARE_ROLLING_GAMES, RECEIVING_POS, RUSHING_POS

class ShareProjectionModel:
    def __init__(self, projection_year: int, seed: int = 42):
        self.projection_year = projection_year
        self.seed = seed
        self.pipelines: Dict[str, Pipeline] = {}
        self.feature_cols: List[str] = []
        self.model_dir = Path(ARTIFACTS_DIR) / SHARE_MODEL_DIRNAME / str(projection_year)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _make_pipeline_targets(self, num: List[str], cat: List[str]) -> Pipeline:
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num),
                ("cat", OneHotEncoder(categories=[RECEIVING_POS], handle_unknown="ignore"), cat),
            ],
            remainder="drop"
        )
        return Pipeline([
            ("pre", pre),
            ("gbr", GradientBoostingRegressor(random_state=self.seed))
        ])

    def _make_pipeline_rush(self, num: List[str], cat: List[str]) -> Pipeline:
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num),
                ("cat", OneHotEncoder(categories=[RUSHING_POS], handle_unknown="ignore"), cat),
            ],
            remainder="drop"
        )
        return Pipeline([
            ("pre", pre),
            ("gbr", GradientBoostingRegressor(random_state=self.seed))
        ])

    def fit(self, player_shares: pd.DataFrame):
        if (player_shares["season"] >= self.projection_year).any():
            raise ValueError("Player-share training data contains projection year; leakage.")

        num = [
            "age",
            "years_exp",
            "num_active_same_pos",
            "is_home",
            "week",
            f"prev_target_share_{SHARE_ROLLING_GAMES}",
            f"prev_rush_attempt_share_{SHARE_ROLLING_GAMES}",
        ]
        cat = ["position"]
        self.feature_cols = num + cat

        rx = player_shares[player_shares["position"].isin(RECEIVING_POS)].copy()
        tx = player_shares[player_shares["position"].isin(RUSHING_POS)].copy()

        X_targets = rx[self.feature_cols]
        y_targets = rx["target_share"].astype(float).values
        pipe_targets = self._make_pipeline_targets(num, cat)
        pipe_targets.fit(X_targets, y_targets)

        X_rush = tx[self.feature_cols]
        y_rush = tx["rush_attempt_share"].astype(float).values
        pipe_rush = self._make_pipeline_rush(num, cat)
        pipe_rush.fit(X_rush, y_rush)

        self.pipelines = {"target_share": pipe_targets, "rush_attempt_share": pipe_rush}
        self._save()

    def predict_shares(self, roster_features: pd.DataFrame) -> pd.DataFrame:
        if not self.pipelines:
            self._load()
        df = roster_features.copy()
        preds_t = df[df["position"].isin(RECEIVING_POS)].copy()
        preds_r = df[df["position"].isin(RUSHING_POS)].copy()
        base_cols = ["season","week","team","player_id","player_name","position"]
        if "position_rank" in df.columns:
            base_cols.append("position_rank")
        out = df[base_cols].copy()
        if not preds_t.empty:
            pt = self.pipelines["target_share"].predict(preds_t[self.feature_cols])
            preds_t["pred_target_share_raw"] = np.clip(pt, 0.0, None)
            out = out.merge(preds_t[["player_id","pred_target_share_raw"]], on="player_id", how="left")
        if not preds_r.empty:
            pr = self.pipelines["rush_attempt_share"].predict(preds_r[self.feature_cols])
            preds_r["pred_rush_attempt_share_raw"] = np.clip(pr, 0.0, None)
            out = out.merge(preds_r[["player_id","pred_rush_attempt_share_raw"]], on="player_id", how="left")
        return out

    def _save(self):
        for name, pipe in self.pipelines.items():
            joblib.dump({"pipe": pipe, "feature_cols": self.feature_cols}, self.model_dir / f"{name}.joblib")

    def _load(self):
        if not self.pipelines:
            for name in ["target_share","rush_attempt_share"]:
                fp = self.model_dir / f"{name}.joblib"
                blob = joblib.load(fp)
                self.pipelines[name] = blob["pipe"]
                self.feature_cols = blob["feature_cols"]