from dataclasses import dataclass
from typing import Tuple

from .enums import Player

Coord = Tuple[int, int]

@dataclass(frozen=True)
class PlacementAction:
    player: Player
    coord: Coord
    worker_index: int

@dataclass
class TurnAction:
    player: "Player"
    worker_index: int
    move_from: Coord
    move_to: Coord
    build_at: Coord
    build_dome: bool = False