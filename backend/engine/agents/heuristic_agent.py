import random
from typing import List, Union

from .base import Agent
from ..state import GameState
from ..enums import Player, GamePhase
from ..game_types import PlacementAction, TurnAction

class HeuristicAgent(Agent):

    # Agent uses heuristic evaluation to choose best move
    # Does this by looking a move ahead and selects action with highest eval
    # Random Placement

    def __init__(self, player: Player = None):
        self.player = player

    def select_action(
        self, state: GameState, actions: List[Union[PlacementAction, TurnAction]]
    ) -> Union[PlacementAction, TurnAction, None]:
        if not actions:
            return None

        if state.phase == GamePhase.PLACEMENT:
            return random.choice(actions)

        best_action = None
        best_score = -float('inf')

        for action in actions:
            next_state = state.apply_turn(action)
            
            score = self._evaluate_state(next_state, state.to_move)

            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    def _evaluate_state(self, state: GameState, player: Player) -> float:

        if state.is_over():
            if state.winner == player:
                return float('inf')
            elif state.winner == player.opponent():
                return -float('inf')
            else:
                return 0


        # Evaluation function

        # Sum of worker heights
        height_score = 0
        for p, worker_index in state.workers:
            if p == player:
                coord = state.workers.get((p, worker_index))
                if coord:
                    height_score += state.board.get_height(coord)

       # Mobility score for player
        original_to_move = state.to_move
        state.to_move = player
        my_mobility = len(state.legal_turn_actions())
        
        # Mobility score for opponent
        state.to_move = player.opponent()
        opponent_mobility = len(state.legal_turn_actions())
        
        state.to_move = original_to_move

        w_height = 3.0
        w_mobility = 0.5
        
        final_score = (w_height * height_score) + (w_mobility * my_mobility) - (w_mobility * opponent_mobility)
        
        return final_score