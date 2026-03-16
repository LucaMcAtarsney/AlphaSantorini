from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import random

from .state import GameState
from .enums import Player
from .game_types import PlacementAction, TurnAction, TOTAL_ACTIONS


DIRS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def dir_index(dr: int, dc: int) -> int:
    return DIRS.index((dr, dc))


class ActionCodec:
    """Encode/decode actions into a fixed index space.

    Indices:
    - [0..49]: placement -> worker_index*25 + r*5 + c
    - [50..305]: turn    -> 50 + (w*128) + (move_dir*16) + (build_dir*2) + dome_flag
    """

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
        # idx in [0..49]
        worker_index = idx // 25
        r25 = idx % 25
        r, c = divmod(r25, 5)
        return PlacementAction(player, (r, c), worker_index)

    @staticmethod
    def decode_action(state: GameState, idx: int) -> TurnAction:
        # idx in [50..305]
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
    def __init__(self, c_puct: float = 1.5, value_fn: Optional[Callable[[GameState, Player], float]] = None):
        self.c_puct = c_puct
        self.value_fn = value_fn

    def run(self, root_state: GameState, num_simulations: int = 200) -> Dict[int, int]:
        root = MCTSNode(state=root_state.clone(), player_to_move=root_state.to_move)
        if not root.is_expanded:
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
                
                if best_action_idx is None:
                    break
                    
                path.append((node, best_action_idx))
                child = node.children[best_action_idx]
                
                if child.child_node is None:
                    if child.next_state is None:
                        child.next_state = self._apply(node.state, child.action_obj)
                    child.child_node = MCTSNode(
                        state=child.next_state,
                        player_to_move=child.next_state.to_move,
                    )
                node = child.child_node

            # Expansion
            if not node.is_expanded:
                self._expand(node)

            # Simulation
            if node.terminal_value is not None:
                v = node.terminal_value
            else:
                v = self._rollout(node)

            # Backpropagation
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
        if not actions:
            node.is_expanded = True
            node.terminal_value = -1.0
            return

        uniform_prior = 1.0 / len(actions)

        for a in actions:
            idx = ActionCodec.encode_index(node.state, a)
            node.children[idx] = MCTSChild(
                prior=uniform_prior,
                action_idx=idx,
                action_obj=a,
            )
        node.is_expanded = True

    def _rollout(self, node: MCTSNode) -> float:

        rollout_state = node.state.clone()
        while not rollout_state.is_over():
            legal_actions = rollout_state.legal_actions()
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            if isinstance(action, PlacementAction):
                rollout_state = rollout_state.apply_placement(action)
            else:
                rollout_state = rollout_state.apply_turn(action)

        if rollout_state.winner is None:
            return 0.0
        
        return 1.0 if rollout_state.winner == node.player_to_move else -1.0

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

    return math.tanh(0.1 * score)
