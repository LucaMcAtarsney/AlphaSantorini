import random
from typing import List, Union

from .base import Agent
from ..state import GameState
from ..enums import Player
from ..game_types import PlacementAction, TurnAction

class RandomAgent(Agent):
    
    def __init__(self, player: Player = None):
        self.player = player

    def select_action(self, state: GameState, actions: List[Union[PlacementAction, TurnAction]]) -> Union[PlacementAction, TurnAction, None]:
        if not actions:
            return None
        return random.choice(actions)