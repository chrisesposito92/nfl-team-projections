from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, RECEIVING_POS, RUSHING_POS


# --------- utils ---------

def _safe_softmax(x: np.ndarray, temperature: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    """Numerically stable softmax with temperature (returns uniform if degenerate)."""
    if x.size == 0:
        return x
    t = max(float(temperature), eps)
    z = (x / t) - np.max(x / t)
    e = np.exp(z)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(x) / len(x)
    return e / s


def _as_float(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float)


@dataclass
class _Priors:
    """
    Heuristic priors by (position, depth_rank starting at 1).
    Used only when we do not have model scores for a player.
    You can tune these once you look at league-wide historical shares by depth.
    """
    # Target priors
    target_wr: List[float] = (0.27, 0.22, 0.17, 0.10, 0.06, 0.03)
    target_te: List[float] = (0.18, 0.10, 0.05)
    target_rb: List[float] = (0.13, 0.07, 0.04)
    # Rush priors
    rush_rb:   List[float] = (0.62, 0.22, 0.09, 0.04)
    rush_qb:   List[float] = (0.10,)  # QB designed+scramble baseline for QB1
    # Small floor for numerical stability
    eps: float = 1e-6


class ShareProjectionModel:
    """
    Compositional share model.

    Guarantees:
      - sum(target_share) over RECEIVING_POS == 1.0 (if at least one eligible player)
      - sum(rush_attempt_share) over RUSHING_POS == 1.0 (if at least one eligible player)

    How it works (two-stage):
      1) Build per-player *scores* for targets and rushes from whatever you have:
           • If columns like 'target_share' / 'rush_attempt_share' already exist, we treat
             them as unnormalized scores (they do NOT need to sum to 1).
           • Otherwise we fall back to depth-chart priors (WR1 > WR2 > WR3, etc.).
      2) Convert scores to probabilities with a temperature-controlled softmax and
         normalize within each eligibility group to enforce the simplex constraint.

    You can later replace the scoring rule with a learned model without changing call sites.
    """

    def __init__(
        self,
        temperature_target: float = 1.0,
        temperature_rush: float = 1.0,
        priors: Optional[_Priors] = None,
        model_dir: Optional[str] = None,
    ) -> None:
        self.temperature_target = float(temperature_target)
        self.temperature_rush = float(temperature_rush)
        self.priors = priors or _Priors()
        self.model_dir = model_dir or os.path.join(ARTIFACTS_DIR, "share_model_comp")

    # ---------- API surface kept for compatibility with earlier code ----------

    @classmethod
    def load_or_init(cls) -> "ShareProjectionModel":
        """Lightweight loader/initializer (no file I/O yet, but keeps parity with older API)."""
        os.makedirs(os.path.join(ARTIFACTS_DIR, "share_model_comp"), exist_ok=True)
        return cls()

    # No-op fit/save/load to remain compatible with earlier training pipeline calls.
    def fit(self, *args, **kwargs) -> "ShareProjectionModel":
        return self

    def save(self, *args, **kwargs) -> None:
        return None

    @classmethod
    def load(cls, *args, **kwargs) -> "ShareProjectionModel":
        return cls.load_or_init()

    # ---------- main method ----------

    def predict_shares(self, roster: pd.DataFrame) -> pd.DataFrame:
        """
        Input
        -----
        roster: one-team, one-week roster frame with at least:
          - 'player_id', 'player_name', 'position'
          - optional: 'pos_rank' (1=starter in that position room)
          - optional: existing columns 'target_share' / 'rush_attempt_share' or
                      'target_share_pred' / 'rush_attempt_share_pred' as *scores*

        Output
        ------
        roster with 'target_share' and 'rush_attempt_share' added/overwritten,
        compositionally normalized within eligibility groups.
        """
        df = roster.copy()

        # Receiving group (RB/WR/TE)
        rx_mask = df["position"].isin(RECEIVING_POS)
        rx_idx = np.flatnonzero(rx_mask.values)

        # Rushing group (RB/QB as per config)
        ru_mask = df["position"].isin(RUSHING_POS)
        ru_idx = np.flatnonzero(ru_mask.values)

        # ---------- build scores for targets ----------
        rx_scores = np.zeros(len(df), dtype=float)
        if rx_idx.size > 0:
            if "target_share" in df.columns and df.loc[rx_mask, "target_share"].notna().any():
                s = _as_float(df.loc[rx_mask, "target_share"].fillna(0.0).values)
            elif "target_share_pred" in df.columns:
                s = _as_float(df.loc[rx_mask, "target_share_pred"].fillna(0.0).values)
            else:
                s = self._depth_prior_scores(df.loc[rx_mask], kind="target")
            rx_scores[rx_idx] = s

        # ---------- build scores for rush attempts ----------
        ru_scores = np.zeros(len(df), dtype=float)
        if ru_idx.size > 0:
            if "rush_attempt_share" in df.columns and df.loc[ru_mask, "rush_attempt_share"].notna().any():
                s = _as_float(df.loc[ru_mask, "rush_attempt_share"].fillna(0.0).values)
            elif "rush_attempt_share_pred" in df.columns:
                s = _as_float(df.loc[ru_mask, "rush_attempt_share_pred"].fillna(0.0).values)
            else:
                s = self._depth_prior_scores(df.loc[ru_mask], kind="rush")
            ru_scores[ru_idx] = s

        # ---------- convert to compositional probabilities ----------
        out = df.copy()

        if rx_idx.size > 0:
            s = np.maximum(rx_scores[rx_idx], 0.0) + self.priors.eps
            probs = _safe_softmax(np.log(s + self.priors.eps), self.temperature_target)
            out.loc[rx_mask, "target_share"] = probs
        else:
            out["target_share"] = 0.0

        if ru_idx.size > 0:
            s = np.maximum(ru_scores[ru_idx], 0.0) + self.priors.eps
            probs = _safe_softmax(np.log(s + self.priors.eps), self.temperature_rush)
            out.loc[ru_mask, "rush_attempt_share"] = probs
        else:
            out["rush_attempt_share"] = 0.0

        # Final exact renormalization (guard against tiny drift)
        if rx_idx.size > 0:
            s = out.loc[rx_mask, "target_share"].to_numpy(dtype=float)
            z = s.sum()
            if z > 0:
                out.loc[rx_mask, "target_share"] = s / z
        if ru_idx.size > 0:
            s = out.loc[ru_mask, "rush_attempt_share"].to_numpy(dtype=float)
            z = s.sum()
            if z > 0:
                out.loc[ru_mask, "rush_attempt_share"] = s / z

        return out

    # ---------- helpers ----------

    def _depth_prior_scores(self, sub: pd.DataFrame, kind: str) -> np.ndarray:
        """
        Map each player's (position, pos_rank) to a heuristic prior score.
        If pos_rank is missing, assume a large number (deeper on the chart).
        """
        pos = sub["position"].astype(str).values
        rank = sub["pos_rank"].fillna(9).to_numpy(dtype=int) if "pos_rank" in sub.columns else np.full(len(sub), 9, int)
        scores = np.zeros(len(sub), dtype=float)

        for i in range(len(sub)):
            p = pos[i]
            r = int(rank[i])
            if kind == "target":
                if p == "WR":
                    scores[i] = self._get_prior(self.priors.target_wr, r)
                elif p == "TE":
                    scores[i] = self._get_prior(self.priors.target_te, r)
                elif p == "RB":
                    scores[i] = self._get_prior(self.priors.target_rb, r)
                else:
                    scores[i] = self.priors.eps
            elif kind == "rush":
                if p == "RB":
                    scores[i] = self._get_prior(self.priors.rush_rb, r)
                elif p == "QB":
                    scores[i] = self._get_prior(self.priors.rush_qb, r)
                else:
                    scores[i] = self.priors.eps
            else:
                scores[i] = self.priors.eps

        # Avoid all-zeros
        if np.sum(scores) <= 0:
            scores = np.ones_like(scores) / max(len(scores), 1)
        return scores

    @staticmethod
    def _get_prior(arr: Iterable[float], rank: int) -> float:
        idx = max(1, rank) - 1
        if idx >= len(arr):
            # geometric decay for tail ranks
            tail = float(arr[-1]) if len(arr) > 0 else 0.01
            return tail * (0.6 ** (idx - (len(arr) - 1)))
        return float(arr[idx])