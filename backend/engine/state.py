from __future__ import annotations
import copy
from typing import List, Dict, Optional, Iterable, Tuple, Union

from .board import Board
from .enums import Player, GamePhase
from .game_types import Coord, PlacementAction, TurnAction

class GameState:

    def __init__(self):

        self.board = Board()
        self.workers: Dict[Player, List[Coord]] = {}
        self.to_move: Player = Player.ONE
        self.phase: GamePhase = GamePhase.PLACEMENT
        self.winner: Optional[Player] = None

    def clone(self) -> GameState:
        
        new_state = GameState()
        new_state.board = self.board.clone()
        new_state.workers = copy.deepcopy(self.workers)
        new_state.to_move = self.to_move
        new_state.phase = self.phase
        new_state.winner = self.winner
        return new_state
    
    def worker_coord(self, player: Player, index: int) -> Coord:
        
        # Returns the coord of worker from dictionary
        return self.workers.get((player, index))
    
    def is_occupied(self, coord: Coord) -> bool:
        
        # Checks if any worker is at coord
        return coord in self.workers.values()
    
    def is_over(self) -> bool:
        return self.phase == GamePhase.GAME_OVER
    
    def legal_actions(self) -> List[Union[PlacementAction, TurnAction]]:

        if self.is_over():
            return []
        
        if self.phase == GamePhase.PLACEMENT:
            return self.legal_placement_actions()
        
        if self.phase == GamePhase.MAIN:
            return self.legal_turn_actions()
        
        return []
    
    def legal_placement_actions(self) -> List[PlacementAction]:

        player = self.to_move
        placed_workers = sum(1 for p, _ in self.workers if p == player)

        if placed_workers >= 2:
            return []
        
        worker_index = placed_workers
        actions = []

        for row in range(5):
            for col in range(5):
                coord = (row, col)
                if not self.is_occupied(coord):
                    actions.append(PlacementAction(player, coord, worker_index))
        
        return actions
    
    def apply_placement(self, action: PlacementAction):

        if self.phase != GamePhase.PLACEMENT:
            raise ValueError("Not in placement phase")
        
        state = self.clone()
        state.workers[(action.player, action.worker_index)] = action.coord

        if len(state.workers) == 4:
            state.phase = GamePhase.MAIN
            state.to_move = Player.ONE
        elif action.worker_index == 1:
            state.to_move = action.player.opponent()
        
        return state
    
    def adjacency_coords(self, coord: Coord) -> Iterable[Coord]:

        row, col = coord

        for deltaRow in [-1, 0, 1]:
            for deltaCol in [-1, 0, 1]:
                
                # Original position
                if deltaRow == 0 and deltaCol == 0:
                    continue

                adj_coord = (row + deltaRow, col + deltaCol)
                if self.board.in_bounds(adj_coord):
                    yield adj_coord

    def legal_turn_actions(self) -> List[TurnAction]:

        # Generate all legal move and build actions for curr player
        actions = []
        player = self.to_move
        for worker_index in [0,1]:
            move_from = self.workers.get((player, worker_index))

            if not move_from:
                continue

            current_height = self.board.get_height(move_from)

            # Check all possible moves
            for move_to in self.adjacency_coords(move_from):

                # Check if move is legal
                is_valid_move = (
                    not self.is_occupied(move_to) and
                    not self.board.is_dome(move_to) and
                    self.board.get_height(move_to) <= current_height + 1
                )

                # Skip if illegal
                if not is_valid_move:
                    continue

                # Temporarily move worker to check for build positions
                original_workers = self.workers.copy()
                self.workers[(player, worker_index)] = move_to

                # Check all possible build locations
                for build_at in self.adjacency_coords(move_to):

                    is_valid_build = (
                        not self.is_occupied(build_at) and
                        not self.board.is_dome(build_at)
                    )

                    if is_valid_build:
                        # Standard build action
                        if self.board.get_height(build_at) < 3:
                            actions.append(TurnAction(
                                player,
                                worker_index,
                                move_from,
                                move_to,
                                build_at,
                                build_dome=False
                            ))
                        
                        # If tower is at level 3 add option to build a dome
                        if self.board.get_height(build_at) == 3:
                            actions.append(TurnAction(
                                player,
                                worker_index,
                                move_from,
                                move_to,
                                build_at,
                                build_dome=True
                            ))

                # Restore original worker position
                self.workers = original_workers

        return actions
    
    def legal_turn_actions_for_player(self, player: Player) -> List[TurnAction]:
        # Returns a list of all legal moves for a specific player
        if self.phase != GamePhase.MAIN:
            return []

        # Temporarily set the to_move player to generate actions
        original_to_move = self.to_move
        self.to_move = player
        
        actions = self.legal_turn_actions()
        
        # Restore the original to_move player
        self.to_move = original_to_move
        
        return actions

    def apply_turn(self, action: TurnAction) -> GameState:

        if self.phase != GamePhase.MAIN:
            raise ValueError("Not in main phase")
        
        state = self.clone()

        # Move worker
        state.workers[(action.player, action.worker_index)] = action.move_to

        # Check for win condition
        if state.board.get_height(action.move_to) == 3:
            state.phase = GamePhase.GAME_OVER
            state.winner = action.player
            return state

        # Build at specified location
        state.board.build(action.build_at, is_dome=action.build_dome)

        # Switch turn to opponent
        state.to_move = action.player.opponent()

        # Check for possible moves for opponent
        if not state.legal_turn_actions():
            state.phase = GamePhase.GAME_OVER
            state.winner = action.player

        return state
    
    def to_dict(self):

        serializable_workers = {
            f"{player.name}_{index}": coord
            for (player, index), coord in self.workers.items()
        }

        return {
            "board": self.board.heights,
            "workers": serializable_workers,
            "phase": self.phase.name,
            "to_move": self.to_move.name,
            "winner": self.winner.name if self.winner else None
        }