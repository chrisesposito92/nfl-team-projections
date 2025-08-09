from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

from .config import QTR_SECONDS

@dataclass
class GameState:
    # Possession & field
    offense: str
    defense: str
    yardline: int          # 1..99 (offense going toward 100)
    down: int              # 1..4
    distance: int          # yards to go
    # Clock & quarter
    quarter: int
    clock: int             # seconds remaining in current quarter
    # Score
    score_off: int
    score_def: int
    # Timeouts (not fully used in MVP)
    to_off: int
    to_def: int
    # Bookkeeping
    play_id: int
    drive_id: int
    # Flags
    is_kickoff: bool = False

    def flip_possession(self, new_yardline: int | None = None) -> None:
        self.offense, self.defense = self.defense, self.offense
        self.score_off, self.score_def = self.score_def, self.score_off
        self.to_off, self.to_def = self.to_def, self.to_off
        self.down = 1
        self.distance = 10
        self.yardline = int(new_yardline if new_yardline is not None else 25)
        self.drive_id += 1

    def first_down_reset(self, new_yardline: int) -> None:
        self.down = 1
        self.distance = 10
        self.yardline = int(new_yardline)

    def advance_down(self, new_yardline: int, gained: int) -> None:
        self.yardline = int(new_yardline)
        if gained >= self.distance:
            self.first_down_reset(new_yardline)
        else:
            self.distance = max(1, self.distance - gained)
            self.down += 1

    def add_points_offense(self, pts: int) -> None:
        self.score_off += int(pts)

    def as_row(self) -> Dict[str, Any]:
        return {
            "qtr": self.quarter,
            "clock": f"{self.clock // 60:02d}:{self.clock % 60:02d}",
            "offense": self.offense,
            "defense": self.defense,
            "down": self.down,
            "dist": self.distance,
            "yardline": self.yardline,
            "drive_id": self.drive_id,
            "play_id": self.play_id,
        }


def start_state(home: str, away: str, receive_first: str = "away") -> GameState:
    # MVP: away receives first; 2nd half flips automatically in engine.
    offense = away if receive_first == "away" else home
    defense = home if offense == away else away
    return GameState(
        offense=offense, defense=defense,
        yardline=25, down=1, distance=10,
        quarter=1, clock=QTR_SECONDS,
        score_off=0, score_def=0,
        to_off=3, to_def=3, play_id=0, drive_id=1,
        is_kickoff=True
    )