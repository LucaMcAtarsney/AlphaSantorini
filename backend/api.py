from fastapi import FastAPI
from engine.state import GameState
from engine.enums import Player
from engine.game_types import PlacementAction, TurnAction
from engine.agents.random_agent import RandomAgent

app = FastAPI()

game = GameState()

agents = {
    Player.ONE: RandomAgent(),
    Player.TWO: RandomAgent(),
}

@app.get("/game")
def get_game_state():
    return game.to_dict()

@app.post("/game/reset")
def reset_game():
    global game
    game = GameState()
    return {"message": "Game reset OK"}

@app.post("/game/step")
def game_step():
    global game
    if not game.is_over():
        agent = agents[game.to_move]
        legal_actions = game.legal_actions()
        if legal_actions:
            action = agent.select_action(game, legal_actions)
            if isinstance(action, PlacementAction):
                game = game.apply_placement(action)
            elif isinstance(action, TurnAction):
                game = game.apply_turn(action)
            return {"action": str(action)}
    return {"message": "Game over/ no actions available"}


   
