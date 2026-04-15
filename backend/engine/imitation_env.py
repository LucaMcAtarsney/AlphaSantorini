import torch
from typing import List, Tuple

from .state import GameState
from .enums import Player
from .game_types import TurnAction, Coord

# N, NE, E, SE, S, SW, W, NW
DIRS: List[Tuple[int, int]] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
]

def dir_to_index(dr: int, dc: int) -> int:

    # Vector to an index  0 to 7
    try:
        return DIRS.index((dr, dc))
    except ValueError:
        raise ValueError(f"Invalid direction: {(dr, dc)}")

def encode_state_tensor(state: GameState, player: Player) -> torch.Tensor:
    """
    Planes:
    0-4: Height 0, 1, 2, 3, 4=dome
    5: Player worker 0
    6: Player worker 1
    7: Opponent worker 0
    8: Opponent worker 1
    9: Side to move 1.0 if player turn
    """
    planes = torch.zeros(10, 5, 5, dtype=torch.float32)
    
    # 0-4
    for r in range(5):
        for c in range(5):
            height = state.board.get_height((r, c))
            planes[height, r, c] = 1.0

    # 5-8
    opponent = player.opponent()
    worker_planes = {
        (player, 0): 5,
        (player, 1): 6,
        (opponent, 0): 7,
        (opponent, 1): 8,
    }
    for (p, worker_idx), coord in state.workers.items():
        if coord:
            plane_idx = worker_planes.get((p, worker_idx))
            if plane_idx is not None:
                planes[plane_idx, coord[0], coord[1]] = 1.0

    # 9
    if state.to_move == player:
        planes[9, :, :] = 1.0
        
    return planes

class ActionCodec128:

    @staticmethod
    def encode(action: TurnAction) -> int:
        # Converts TurnAction to 128 index
        worker_id = action.worker_index
        
        dr_m, dc_m = action.move_to[0] - action.move_from[0], action.move_to[1] - action.move_from[1]
        move_dir = dir_to_index(dr_m, dc_m)
        
        dr_b, dc_b = action.build_at[0] - action.move_to[0], action.build_at[1] - action.move_to[1]
        build_dir = dir_to_index(dr_b, dc_b)
        
        return worker_id * 64 + move_dir * 8 + build_dir

    @staticmethod
    def decode(action_idx: int, state: GameState, player: Player) -> TurnAction:
        # Converts 128 index to TurnAction
        worker_id = action_idx // 64
        rem = action_idx % 64
        move_dir_idx = rem // 8
        build_dir_idx = rem % 8

        move_from = state.worker_coord(player, worker_id)
        if not move_from:
            raise ValueError(f"Cannot decode action, player {player} worker {worker_id} not on board.")

        dr_m, dc_m = DIRS[move_dir_idx]
        move_to = (move_from[0] + dr_m, move_from[1] + dc_m)

        dr_b, dc_b = DIRS[build_dir_idx]
        build_at = (move_to[0] + dr_b, move_to[1] + dc_b)
        
        return TurnAction(player, worker_id, move_from, move_to, build_at)

def get_legal_action_mask(state: GameState, player: Player) -> torch.Tensor:

    # Returns a boolean tensor of 128 where True = legal action
    mask = torch.zeros(128, dtype=torch.bool)
    legal_actions = state.legal_actions()
    
    turn_actions = [action for action in legal_actions if isinstance(action, TurnAction)]

    for action in turn_actions:
        if action.build_dome:
            continue
        
        try:
            action_idx = ActionCodec128.encode(action)
            mask[action_idx] = True
        except (ValueError, IndexError):
            continue
            
    return mask