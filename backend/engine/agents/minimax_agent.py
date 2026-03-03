import random
from typing import List, Union, Optional

from .base import Agent
from ..state import GameState
from ..enums import Player
from ..game_types import PlacementAction, TurnAction


class MinimaxAgent(Agent):

    # Uses evaluation function
    # Random Placement
    # Follows Minimax
    
    def __init__(self, player: Player, depth: int = 3):
        self.player = player
        self.depth = depth

    def _evaluate_board(self, state: GameState) -> float:

        if state.winner:
            if state.winner == self.player:
                return float('inf')
            elif state.winner == self.player.opponent():
                return float('-inf')
            else:
                return 0

        score = 0.0
        
        # Height Score
        for worker_index in [0, 1]:
            player_worker = state.worker_coord(self.player, worker_index)
            opp_worker = state.worker_coord(self.player.opponent(), worker_index)
            if player_worker:
                score += state.board.get_height(player_worker) * 10
            if opp_worker:
                score -= state.board.get_height(opp_worker) * 10

        # Mobility score
        player_actions = state.legal_turn_actions_for_player(self.player)
        opp_actions = state.legal_turn_actions_for_player(self.player.opponent())
        score += len(player_actions)
        score -= len(opp_actions)

        return score
    
    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximising_player: bool) -> float:
        if depth == 0 or state.is_over():
            return self._evaluate_board(state)
        
        legal_actions = state.legal_turn_actions()
        if not legal_actions:
            return self._evaluate_board(state)
        
        if maximising_player:
            max_eval = float('-inf')
            for action in legal_actions:
                next_state = state.apply_turn(action)
                eval = self._minimax(next_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for action in legal_actions:
                next_state = state.apply_turn(action)
                eval = self._minimax(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def select_action(
        self,
        state: GameState,
        actions: List[Union[PlacementAction,TurnAction]]
        ) -> Optional[Union[PlacementAction, TurnAction]]:
        if not actions:
            return None
        
        if isinstance(actions[0], PlacementAction):
            best_action = None
            max_options = -1
            for action in actions:
                temp_state = state.apply_placement(action)

                if temp_state.to_move == self.player:
                    return random.choice(actions)
                
                opp_actions = temp_state.legal_placement_actions()
                if len(opp_actions) > max_options:
                    max_options = len(opp_actions)
                    best_action = action

            return best_action or random.choice(actions)
        
        best_action = None
        best_value = float('-inf')

        for action in actions:
            next_state = state.apply_turn(action)
            board_value = self._minimax(next_state, self.depth - 1, float('-inf'), float('inf'), False)

            if board_value > best_value:
                best_value = board_value
                best_action = action
        
        return best_action or random.choice(actions)