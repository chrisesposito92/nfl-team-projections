# src/pbp/players.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd

from ..config import OFFENSE_POS, DEPTH_CHART_LIMITS, RECEIVING_POS, RUSHING_POS
from ..data import load_depth_charts
from ..models_shares import ShareProjectionModel
from .tilts import load_ngs_tilts_for_game, _compose_base_shares


@dataclass
class TeamSampling:
    team: str
    rx_df: pd.DataFrame  # columns: player_id, player_name, position, prob
    ru_df: pd.DataFrame  # columns: player_id, player_name, position, prob
    qb1_id: Optional[str]


@dataclass
class GameSampling:
    home: TeamSampling
    away: TeamSampling


def _depth_chart_roster(team: str, year: int) -> pd.DataFrame:
    """Lightweight roster from latest depth chart; limited to offense positions."""
    dc = load_depth_charts([year])
    dc = dc[dc["team"] == team].copy()
    if "dt" in dc.columns:
        dc["dt"] = pd.to_datetime(dc["dt"], errors="coerce")
        dc = dc[dc["dt"] == dc["dt"].max()].copy()

    dc = dc[dc["pos_abb"].isin(OFFENSE_POS)].copy()
    dc["pos_rank"] = pd.to_numeric(dc["pos_rank"], errors="coerce")
    dc = dc[dc["pos_rank"].notna()].copy()
    dc["pos_rank"] = dc["pos_rank"].astype(int)

    def _take_top(g: pd.DataFrame) -> pd.DataFrame:
        lim = int(DEPTH_CHART_LIMITS.get(g.name, 0))
        if lim <= 0:
            return g.iloc[0:0]
        return g.nsmallest(lim, "pos_rank")

    dc = dc.groupby("pos_abb", group_keys=False).apply(_take_top)
    dc["player_id"] = dc["gsis_id"].where(dc["gsis_id"].notna(), dc["player_name"])

    roster = dc.rename(columns={"pos_abb": "position"})[
        ["player_id", "player_name", "position", "pos_rank"]
    ].copy()
    roster = roster.rename(columns={"pos_rank": "position_rank"})

    # Minimal columns to keep ShareProjectionModel happy
    roster["season"] = year
    roster["week"] = 0
    roster["team"] = team
    roster["age"] = 0.0
    roster["years_exp"] = 0.0
    return roster.reset_index(drop=True)


def _build_team_sampling(team: str, year: int) -> TeamSampling:
    roster = _depth_chart_roster(team, year)
    share_model = ShareProjectionModel.load_or_init()
    shares = share_model.predict_shares(roster)

    rx_df = shares[shares["position"].isin(RECEIVING_POS)][
        ["player_id", "player_name", "position", "target_share"]
    ].copy()
    rx_df = rx_df.rename(columns={"target_share": "prob"}).reset_index(drop=True)

    ru_df = shares[shares["position"].isin(RUSHING_POS)][
        ["player_id", "player_name", "position", "rush_attempt_share"]
    ].copy()
    ru_df = ru_df.rename(columns={"rush_attempt_share": "prob"}).reset_index(drop=True)

    qb = shares.loc[shares["position"] == "QB", ["player_id", "position_rank"]].copy()
    qb = qb.sort_values("position_rank").head(1)
    qb1_id = None if qb.empty else str(qb.iloc[0]["player_id"])

    # Final exact normalization, just in case
    if len(rx_df) > 0:
        s = float(rx_df["prob"].sum())
        if s > 0:
            rx_df["prob"] = rx_df["prob"] / s
    if len(ru_df) > 0:
        s = float(ru_df["prob"].sum())
        if s > 0:
            ru_df["prob"] = ru_df["prob"] / s

    return TeamSampling(team=team, rx_df=rx_df, ru_df=ru_df, qb1_id=qb1_id)


def build_game_sampling(home: str, away: str, year: int) -> GameSampling:
    return GameSampling(
        home=_build_team_sampling(home, year),
        away=_build_team_sampling(away, year),
    )


def _init_stat_row() -> Dict[str, float]:
    return {
        "targets": 0.0,
        "receptions": 0.0,
        "rec_yards": 0.0,
        "rec_tds": 0.0,
        "rush_att": 0.0,
        "rush_yards": 0.0,
        "rush_tds": 0.0,
        "pass_yards": 0.0,
        "pass_tds": 0.0,
    }


def _get_qb_meta(ts: TeamSampling) -> Tuple[Optional[str], Optional[str]]:
    """Return (qb1_id, qb1_name) using any known row in rx/ru frames."""
    if ts.qb1_id is None:
        return None, None
    qb1_id = ts.qb1_id
    for df in (ts.ru_df, ts.rx_df):
        if df is not None and len(df) > 0:
            m = df[df["player_id"] == qb1_id]
            if not m.empty:
                return qb1_id, str(m.iloc[0]["player_name"])
    return qb1_id, "QB1"

def attribute_players_from_plays(
    plays: pd.DataFrame,
    home: str,
    away: str,
    year: int,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Convert simulated plays into player-level box scores.
    Uses: offense, play_type, yards_gained, points, is_complete, is_sack.
    Falls back to 'result' string if flags are missing.
    Also attributes QB pass_yards / pass_tds to QB1.
    """
    rng = np.random.RandomState(seed)

    # Base shares + NGS tilts for target/carry selection
    home_tgt, home_ru, home_meta = _compose_base_shares(home, year)
    away_tgt, away_ru, away_meta = _compose_base_shares(away, year)
    tilts = load_ngs_tilts_for_game(home, away, year)

    # Find QB1s for pass attribution
    gs = build_game_sampling(home, away, year)
    home_qb1_id, home_qb1_name = _get_qb_meta(gs.home)
    away_qb1_id, away_qb1_name = _get_qb_meta(gs.away)

    def _weights(ids, base_shares: Dict[str, float], tilt_map: Dict[str, float], floor=1e-9):
        w = []
        for pid in ids:
            b = max(base_shares.get(pid, 0.0), 0.0)
            t = float(tilt_map.get(pid, 1.0))
            w.append(max(b * t, floor))
        s = float(sum(w))
        if s <= 0 or len(ids) == 0:
            return np.array([])
        return np.array([wi / s for wi in w])

    def _elig(meta: Dict[str, Tuple[str, str]], pos_set: Iterable[str]) -> Tuple[list, list, list]:
        ids, names, poss = [], [], []
        for pid, (nm, pos) in meta.items():
            if pos in pos_set:
                ids.append(pid); names.append(nm); poss.append(pos)
        return ids, names, poss

    home_rx_ids, home_rx_names, home_rx_pos = _elig(home_meta, RECEIVING_POS)
    away_rx_ids, away_rx_names, away_rx_pos = _elig(away_meta, RECEIVING_POS)
    home_ru_ids, home_ru_names, home_ru_pos = _elig(home_meta, RUSHING_POS)
    away_ru_ids, away_ru_names, away_ru_pos = _elig(away_meta, RUSHING_POS)

    agg: Dict[tuple, Dict[str, float]] = {}
    def _ensure(team, pid, name, pos):
        key = (team, pid)
        if key not in agg:
            agg[key] = {
                "team": team,
                "player_id": pid,
                "player_name": name,
                "position": pos,
                "targets": 0.0,
                "receptions": 0.0,
                "rec_yards": 0.0,
                "rec_tds": 0.0,
                "rush_att": 0.0,
                "rush_yards": 0.0,
                "rush_tds": 0.0,
                "pass_yards": 0.0,
                "pass_tds": 0.0,
            }
        return key

    def _bump(team, pid, name, pos, field, val):
        key = _ensure(team, pid, name, pos)
        agg[key][field] += float(val)

    for r in plays.itertuples(index=False):
        team = getattr(r, "offense")
        ptype = getattr(r, "play_type")
        yards = float(getattr(r, "yards_gained", 0.0) or 0.0)
        pts = float(getattr(r, "points", 0.0) or 0.0)

        # robust flag fallback via `result` if explicit fields missing
        result = str(getattr(r, "result", "")).upper()
        is_complete = bool(getattr(r, "is_complete", None))
        if is_complete is None:
            is_complete = (result in ("COMPLETE", "TD"))
        is_sack = bool(getattr(r, "is_sack", None))
        if is_sack is None:
            is_sack = (result == "SACK")

        # who is QB1 for this team?
        qb_id, qb_name = (home_qb1_id, home_qb1_name) if team == home else (away_qb1_id, away_qb1_name)
        if qb_id:
            _ensure(team, qb_id, qb_name or "QB1", "QB")

        if ptype == "pass":
            # Assign QB stats regardless of receiver selection
            if qb_id:
                _bump(team, qb_id, qb_name or "QB1", "QB", "pass_yards", max(0.0, yards))
                if pts >= 6.0:
                    _bump(team, qb_id, qb_name or "QB1", "QB", "pass_tds", 1.0)

            if is_sack:
                continue

            if team == home:
                ids, names, poss = home_rx_ids, home_rx_names, home_rx_pos
                base, tilt_tgt = home_tgt, tilts["rec_target"][home]
            else:
                ids, names, poss = away_rx_ids, away_rx_names, away_rx_pos
                base, tilt_tgt = away_tgt, tilts["rec_target"][away]

            if len(ids) == 0:
                continue
            w = _weights(ids, base, tilt_tgt)
            if w.size == 0:
                continue
            i = int(rng.choice(len(ids), p=w))
            pid, nm, pos = ids[i], names[i], poss[i]

            _bump(team, pid, nm, pos, "targets", 1.0)
            if is_complete:
                _bump(team, pid, nm, pos, "receptions", 1.0)
                _bump(team, pid, nm, pos, "rec_yards", max(0.0, yards))
                if pts >= 6.0:
                    _bump(team, pid, nm, pos, "rec_tds", 1.0)

        elif ptype == "run":
            if team == home:
                ids, names, poss = home_ru_ids, home_ru_names, home_ru_pos
                base, tilt_run = home_ru, tilts["rush_carry"][home]
            else:
                ids, names, poss = away_ru_ids, away_ru_names, away_ru_pos
                base, tilt_run = away_ru, tilts["rush_carry"][away]

            if len(ids) == 0:
                continue
            w = _weights(ids, base, tilt_run)
            if w.size == 0:
                continue
            i = int(rng.choice(len(ids), p=w))
            pid, nm, pos = ids[i], names[i], poss[i]

            _bump(team, pid, nm, pos, "rush_att", 1.0)
            _bump(team, pid, nm, pos, "rush_yards", yards)
            if pts >= 6.0:
                _bump(team, pid, nm, pos, "rush_tds", 1.0)

        else:
            continue

    out = pd.DataFrame(list(agg.values()))
    if out.empty:
        out = pd.DataFrame(columns=[
            "team","player_id","player_name","position",
            "targets","receptions","rec_yards","rec_tds",
            "rush_att","rush_yards","rush_tds",
            "pass_yards","pass_tds",
        ])
    return out.sort_values(["team","position","player_name"]).reset_index(drop=True)