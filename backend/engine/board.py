from __future__ import annotations
import copy
from .game_types import Coord

class Board:
    def __init__(self):

        # Set up 5x5 board with heights of 0
        self.heights = [[0 for _ in range(5)] for _ in range(5)]

    def clone(self) -> Board:
        
        # Returns deep copy of board for checking possible moves
        new_board = Board()
        new_board.heights = copy.deepcopy(self.heights)
        return new_board
    
    def in_bounds(self, coord: Coord) -> bool:
    
        # Checks if given coord is within 5x5 board
        row, col = coord
        return 0 <= row < 5 and 0 <= col < 5
    
    def get_height(self, coord: Coord) -> int:
        
        # Returns height of building at given coord
        row, col = coord
        return self.heights[row][col]
    
    def is_dome(self, coord: Coord) -> bool:
        
        # Checks for dome at coord
        return self.get_height(coord) == 4
    
    def build(self, coord: Coord, is_dome: bool = False):
        
        # Builds a block or dome at given coord
        if not self.in_bounds(coord):
            raise ValueError("Build coordinate out of bounds")
        
        current_height = self.get_height(coord)
        if self.is_dome(coord):
            raise ValueError("Cannot build on a dome")

        if is_dome:
            if current_height != 3:
                raise ValueError("Domes can only be built on level 3 towers")
            self.heights[coord[0]][coord[1]] = 4
        else:
            if current_height >= 3:
                raise ValueError("Cannot build higher than level 3 without building a dome")
            self.heights[coord[0]][coord[1]] += 1