from abc import ABC, abstractmethod
from typing import List, Union

from ..state import GameState
from ..game_types import PlacementAction, TurnAction

class Agent(ABC):

    @abstractmethod
    def select_action(self, state: GameState, actions: List[Union[PlacementAction, TurnAction]]) -> Union[PlacementAction, TurnAction, None]:
        
        pass