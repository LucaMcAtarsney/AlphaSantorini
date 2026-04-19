import os
import random
from typing import List, Union, Optional

import torch
import numpy as np

from .base import Agent
from ..state import GameState
from ..enums import Player
from ..game_types import PlacementAction, TurnAction
from ..mcts import MCTS, VanillaMCTS, ActionCodec, heuristic_value
from ..nn.model import PolicyValueNet

DEFAULT_CHECKPOINT_PATH = os.path.join("checkpoints", "koalemos.pth")


class MCTSAgent(Agent):
    # Agent that uses Monte Carlo Tree Search guided by Network

    def __init__(
        self,
        player: Player,
        net: PolicyValueNet,
        sims: int = 400,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        device: str = "cpu",
        use_heuristic: bool = False,
    ):
        self.player = player
        self.sims = sims
        self.temperature = temperature
        
        value_fn = heuristic_value if use_heuristic else None
        self.mcts = MCTS(net, c_puct=c_puct, device=device, value_fn=value_fn)

    @classmethod
    def from_checkpoint(
        cls,
        player: Player,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        sims: int = 400,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        device: str = "auto",
        use_heuristic: bool = False,
    ) -> "MCTSAgent":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        net = PolicyValueNet()
        if os.path.exists(checkpoint_path):
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Warning: checkpoint '{checkpoint_path}' not found, using untrained network.")
        net.to(device)
        net.eval()
        return cls(player=player, net=net, sims=sims, c_puct=c_puct,
                   temperature=temperature, device=device, use_heuristic=use_heuristic)

    def select_action(
        self,
        state: GameState,
        actions: List[Union[PlacementAction, TurnAction]]
    ) -> Optional[Union[PlacementAction, TurnAction]]:
        
        if not actions:
            return None

        # Run MCTS simulations
        visit_counts = self.mcts.run(state, num_simulations=self.sims)
        
        if not visit_counts:
            return random.choice(actions)

        action_indices = list(visit_counts.keys())
        counts = np.array(list(visit_counts.values()), dtype=np.float32)

        if self.temperature == 0.0:
            best_action_idx = action_indices[np.argmax(counts)]
        else:
            dist = counts**(1.0 / self.temperature)
            dist /= np.sum(dist)
            choice = np.random.choice(len(action_indices), p=dist)
            best_action_idx = action_indices[choice]

        if best_action_idx < 50:
            return ActionCodec.placement_from_index(state.to_move, best_action_idx)
        else:
            for action in actions:
                if ActionCodec.encode_index(state, action) == best_action_idx:
                    return action
        
        return random.choice(actions)


class VanillaMCTSAgent(Agent):
    # MCTS agent using uniform priors and random rollouts

    def __init__(self, player: Player, sims: int = 400, c_puct: float = 1.4, temperature: float = 0.0):
        self.player = player
        self.sims = sims
        self.temperature = temperature
        self.mcts = VanillaMCTS(c_puct=c_puct)

    def select_action(
        self,
        state: GameState,
        actions: List[Union[PlacementAction, TurnAction]]
    ) -> Optional[Union[PlacementAction, TurnAction]]:
        if not actions:
            return None

        visit_counts = self.mcts.run(state, num_simulations=self.sims)

        if not visit_counts:
            return random.choice(actions)

        action_indices = list(visit_counts.keys())
        counts = np.array(list(visit_counts.values()), dtype=np.float32)

        if self.temperature == 0.0:
            best_action_idx = action_indices[np.argmax(counts)]
        else:
            dist = counts ** (1.0 / self.temperature)
            dist /= np.sum(dist)
            choice = np.random.choice(len(action_indices), p=dist)
            best_action_idx = action_indices[choice]

        if best_action_idx < 50:
            return ActionCodec.placement_from_index(state.to_move, best_action_idx)
        else:
            for action in actions:
                if ActionCodec.encode_index(state, action) == best_action_idx:
                    return action

        return random.choice(actions)
