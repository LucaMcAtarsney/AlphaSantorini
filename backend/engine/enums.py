from enum import Enum, auto

class Player(Enum):

    ONE = auto()
    TWO = auto()

    def opponent(self):
        if self == Player.ONE:
            return Player.TWO
        else:
            return Player.ONE
        
class GamePhase(Enum):
    
    PLACEMENT = auto()
    MAIN = auto()
    GAME_OVER = auto()