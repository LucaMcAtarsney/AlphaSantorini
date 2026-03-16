from .base import Agent
from ..state import GameState
from ..mcts import MCTS, ActionCodec
import numpy as np

class MCTSAgent(Agent):
    def __init__(self, player, num_simulations: int = 200, c_puct: float = 1.5):
        self.player = player
        self.num_simulations = num_simulations
        self.mcts = MCTS(c_puct=c_puct)

    def select_action(self, game: GameState, legal_actions):
        if not legal_actions:
            return None

        visit_counts = self.mcts.run(game, num_simulations=self.num_simulations)

        if not visit_counts:
            return np.random.choice(legal_actions)

        action_indices = list(visit_counts.keys())
        counts = np.array([visit_counts[i] for i in action_indices])
        probs = counts / np.sum(counts)

        chosen_index = np.random.choice(action_indices, p=probs)

        if chosen_index < 50:
            return ActionCodec.placement_from_index(game.to_move, chosen_index)
        else:
            return ActionCodec.decode_action(game, chosen_index)

    def __str__(self):
        return f"MCTSAgent(sims={self.num_simulations})"
