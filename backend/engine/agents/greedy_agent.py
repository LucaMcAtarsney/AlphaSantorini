from typing import List, Union
import random

from engine.state import GameState
from engine.enums import Player, GamePhase
from engine.game_types import TurnAction, PlacementAction
from .base import Agent

class GreedyAgent(Agent):

    # Prioritises moving up in height
    # If more than 1 upward moves is possible it will pick randomly
    # If no upwards moves, moves randomly
    # Random placement

    def __init__(self, player: Player = None):
        self.player = player

    def select_action(
        self, state: GameState, actions: List[Union[PlacementAction, TurnAction]]
        ) -> Union[PlacementAction, TurnAction, None]:
        
        if not actions:
            return None

        if state.phase == GamePhase.PLACEMENT:
            return random.choice(actions)

        board = state.board
        
        upward_moves = []
        for action in actions:
            if isinstance(action, TurnAction):
                current_height = board.get_height(action.move_from)
                next_height = board.get_height(action.move_to)
                if next_height > current_height:
                    upward_moves.append(action)
        
        if upward_moves:
            return random.choice(upward_moves)
        
        return random.choice(actions)