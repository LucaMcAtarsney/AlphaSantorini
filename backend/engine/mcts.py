from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import random

import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet

from .state import GameState
from .enums import Player
from .game_types import PlacementAction, TurnAction, TOTAL_ACTIONS
from .nn.model import PolicyValueNet

DIRS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def dir_index(dr: int, dc: int) -> int:
    return DIRS.index((dr, dc))


class ActionCodec:
    # Encode/decode actions fixed index space
    @staticmethod
    def encode_index(state: GameState, action) -> int:
        if isinstance(action, PlacementAction):
            (r, c) = action.coord
            return action.worker_index * 25 + r * 5 + c
        else:
            dr_m = action.move_to[0] - action.move_from[0]
            dc_m = action.move_to[1] - action.move_from[1]
            mdir = dir_index(dr_m, dc_m)

            dr_b = action.build_at[0] - action.move_to[0]
            dc_b = action.build_at[1] - action.move_to[1]
            bdir = dir_index(dr_b, dc_b)

            w = action.worker_index
            dome = 1 if action.build_dome else 0
            return 50 + (w * 128) + (mdir * 16) + (bdir * 2) + dome

    @staticmethod
    def placement_from_index(player: Player, idx: int) -> PlacementAction:
        # idx in 0-49
        worker_index = idx // 25
        r25 = idx % 25
        r, c = divmod(r25, 5)
        return PlacementAction(player, (r, c), worker_index)

    @staticmethod
    def decode_action(state: GameState, idx: int) -> TurnAction:
        # idx in 50-305
        idx -= 50
        w = idx // 128
        rem = idx % 128
        mdir = rem // 16
        rem = rem % 16
        bdir = rem // 2
        dome = rem % 2

        dr_m, dc_m = DIRS[mdir]
        dr_b, dc_b = DIRS[bdir]

        move_from = state.workers[(state.to_move, w)]
        move_to = (move_from[0] + dr_m, move_from[1] + dc_m)
        build_at = (move_to[0] + dr_b, move_to[1] + dc_b)

        return TurnAction(
            player=state.to_move,
            move_from=move_from,
            move_to=move_to,
            build_at=build_at,
            build_dome=bool(dome),
            worker_index=w,
        )

def encode_state_planes(state: GameState) -> torch.Tensor:
    # Encode GameState into 10 planes -  heights, workers, to-move
    planes = torch.zeros(10, 5, 5, dtype=torch.float32)

    for r in range(5):
        for c in range(5):
            h = state.board.get_height((r, c))
            h = max(0, min(4, h))
            planes[h, r, c] = 1.0

    # Workers
    idx_map = {
        (Player.ONE, 0): 5,
        (Player.ONE, 1): 6,
        (Player.TWO, 0): 7,
        (Player.TWO, 1): 8,
    }
    for (p_idx, coord) in state.workers.items():
        p, w_idx = p_idx
        pr = idx_map[(p, w_idx)]
        planes[pr, coord[0], coord[1]] = 1.0

    if state.to_move == Player.ONE:
        planes[9, :, :] = 1.0
    else:
        planes[9, :, :] = 0.0
    return planes


def legal_action_mask(state: GameState, actions: List) -> torch.Tensor:
    mask = torch.zeros(TOTAL_ACTIONS, dtype=torch.bool)
    for a in actions:
        idx = ActionCodec.encode_index(state, a)
        mask[idx] = True
    return mask

@dataclass
class MCTSChild:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    next_state: Optional[GameState] = None
    child_node: Optional["MCTSNode"] = None
    action_idx: int = -1
    action_obj: object = None

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else (self.value_sum / self.visits)


@dataclass
class MCTSNode:
    state: GameState
    player_to_move: Player
    children: Dict[int, MCTSChild] = field(default_factory=dict)
    visits: int = 0
    is_expanded: bool = False
    terminal_value: Optional[float] = None


class MCTS:
    def __init__(self, net: PolicyValueNet, c_puct: float = 1.5, device: str = "cpu", value_fn: Optional[Callable[[GameState, Player], float]] = None):
        self.net = net.to(device)
        self.c_puct = c_puct
        self.device = device
        self.value_fn = value_fn

    def run(self, root_state: GameState, num_simulations: int = 200, root_dirichlet: Optional[Tuple[float, float]] = None) -> Dict[int, int]:
        root = MCTSNode(state=root_state.clone(), player_to_move=root_state.to_move)
        self._expand(root)

        # Optional  noise to tweak exploration during sp
        if root_dirichlet is not None and root.children:
            epsilon, alpha = root_dirichlet

            child_indices = list(root.children.keys())
            K = len(child_indices)
            noise = Dirichlet(torch.full((K,), alpha)).sample().tolist()

            for i, a_idx in enumerate(child_indices):
                ch = root.children[a_idx]
                ch.prior = (1.0 - epsilon) * ch.prior + epsilon * float(noise[i])

        for _ in range(num_simulations):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            # Selection
            while node.is_expanded and node.terminal_value is None and node.children:
                total_N = sum(child.visits for child in node.children.values())
                best_score = -1e9
                best_action_idx = None
                for a_idx, child in node.children.items():
                    u = self.c_puct * child.prior * math.sqrt(total_N + 1e-8) / (1 + child.visits)
                    score = child.q + u
                    if score > best_score:
                        best_score = score
                        best_action_idx = a_idx
                path.append((node, best_action_idx))
                child = node.children[best_action_idx]

                if child.child_node is None:
                    if child.next_state is None:
                        child.next_state = self._apply(node.state, child.action_obj)
                    child.child_node = MCTSNode(
                        state=child.next_state,
                        player_to_move=child.next_state.to_move,
                    )
                    self._expand(child.child_node)
                node = child.child_node

            # Evaluate
            if node.terminal_value is not None:
                v = node.terminal_value
            else:
                v = self._value(node)

            # Backup flip sign each ply
            for (parent, a_idx) in reversed(path):
                ch = parent.children[a_idx]
                ch.visits += 1
                ch.value_sum += v
                parent.visits += 1
                v = -v

        visit_counts: Dict[int, int] = {a_idx: ch.visits for a_idx, ch in root.children.items()}
        return visit_counts

    def _expand(self, node: MCTSNode):
        if node.state.is_over():
            node.is_expanded = True
            if node.state.winner is None:
                node.terminal_value = 0.0
            else:
                node.terminal_value = 1.0 if node.state.winner == node.player_to_move else -1.0
            return

        actions = node.state.legal_actions()
        mask = legal_action_mask(node.state, actions).to(self.device)

        with torch.no_grad():
            x = encode_state_planes(node.state).unsqueeze(0).to(self.device)
            logits, _ = self.net(x)
            logits = logits.squeeze(0)
            logits[~mask] = -1e9
            priors = F.softmax(logits, dim=-1).cpu()

        for a in actions:
            idx = ActionCodec.encode_index(node.state, a)
            node.children[idx] = MCTSChild(
                prior=float(priors[idx].item()),
                action_idx=idx,
                action_obj=a,
            )
        node.is_expanded = True

    def _value(self, node: MCTSNode) -> float:
        # Heuristic value override if provided
        if self.value_fn is not None:
            return float(self.value_fn(node.state, node.player_to_move))
        
        # Default: NN value
        with torch.no_grad():
            x = encode_state_planes(node.state).unsqueeze(0).to(self.device)
            _, value = self.net(x)
            v = float(value.item())
        return v

    @staticmethod
    def _apply(state: GameState, action) -> GameState:
        if isinstance(action, PlacementAction):
            return state.apply_placement(action)
        else:
            return state.apply_turn(action)

class VanillaMCTS:
    # MCTS with uniform priors and random rollouts (no nn)

    def __init__(self, c_puct: float = 1.4):
        self.c_puct = c_puct

    def run(self, root_state: GameState, num_simulations: int = 200) -> Dict[int, int]:
        import random as _rand
        root = MCTSNode(state=root_state.clone(), player_to_move=root_state.to_move)
        self._expand(root)

        for _ in range(num_simulations):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            # Selection
            while node.is_expanded and node.terminal_value is None and node.children:
                total_N = sum(child.visits for child in node.children.values())
                best_score = -1e9
                best_action_idx = None
                for a_idx, child in node.children.items():
                    u = self.c_puct * child.prior * math.sqrt(total_N + 1e-8) / (1 + child.visits)
                    score = child.q + u
                    if score > best_score:
                        best_score = score
                        best_action_idx = a_idx
                path.append((node, best_action_idx))
                child = node.children[best_action_idx]
                if child.child_node is None:
                    if child.next_state is None:
                        child.next_state = self._apply(node.state, child.action_obj)
                    child.child_node = MCTSNode(
                        state=child.next_state,
                        player_to_move=child.next_state.to_move,
                    )
                    self._expand(child.child_node)
                node = child.child_node

            # Random rollout
            if node.terminal_value is not None:
                v = node.terminal_value
            else:
                v = self._rollout(node.state, node.player_to_move)

            # Backup
            for (parent, a_idx) in reversed(path):
                ch = parent.children[a_idx]
                ch.visits += 1
                ch.value_sum += v
                parent.visits += 1
                v = -v

        return {a_idx: ch.visits for a_idx, ch in root.children.items()}

    def _expand(self, node: MCTSNode):
        if node.state.is_over():
            node.is_expanded = True
            if node.state.winner is None:
                node.terminal_value = 0.0
            else:
                node.terminal_value = 1.0 if node.state.winner == node.player_to_move else -1.0
            return

        actions = node.state.legal_actions()
        uniform_prior = 1.0 / len(actions)
        for a in actions:
            idx = ActionCodec.encode_index(node.state, a)
            node.children[idx] = MCTSChild(
                prior=uniform_prior,
                action_idx=idx,
                action_obj=a,
            )
        node.is_expanded = True

    @staticmethod
    def _rollout(state: GameState, perspective: Player) -> float:
        # Random playout to terminal state 
        import random as _rand
        s = state.clone()
        for _ in range(200):
            if s.is_over():
                break
            actions = s.legal_actions()
            if not actions:
                break
            a = _rand.choice(actions)
            if isinstance(a, PlacementAction):
                s = s.apply_placement(a)
            else:
                s = s.apply_turn(a)
        if s.winner is None:
            return 0.0
        return 1.0 if s.winner == perspective else -1.0

    @staticmethod
    def _apply(state: GameState, action) -> GameState:
        if isinstance(action, PlacementAction):
            return state.apply_placement(action)
        else:
            return state.apply_turn(action)


def heuristic_value(state: GameState, player: Player) -> float:
    if state.is_over():
        if state.winner is None:
            return 0.0
        return 1.0 if state.winner == player else -1.0

    height_score = 0
    for (p, w_idx), coord in state.workers.items():
        if p == player and coord is not None:
            height_score += state.board.get_height(coord)

    s = state.clone()
    s.to_move = player
    my_mob = len(s.legal_turn_actions())
    s.to_move = player.opponent()
    opp_mob = len(s.legal_turn_actions())

    score = 2.0 * height_score + 2.0 * my_mob - 2.0 * opp_mob

    # Squash to [-1,1]
    return math.tanh(0.1 * score)
