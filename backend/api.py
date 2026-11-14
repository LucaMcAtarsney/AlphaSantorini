from typing import Dict, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.agents.greedy_agent import GreedyAgent
from engine.agents.heuristic_agent import HeuristicAgent
from engine.agents.minimax_agent import MinimaxAgent
from engine.agents.random_agent import RandomAgent
from engine.enums import Player
from engine.game_types import PlacementAction, TurnAction
from engine.state import GameState


AgentKind = Literal[
    "human",
    "random",
    "greedy",
    "heuristic",
    "minimax-d2",
    "vanilla-mcts",
    "mcts-nn",
    "mcts-heuristic",
]

MCTS_SIMULATIONS = 200
DEFAULT_CHECKPOINT_PATH = "checkpoints/koalemos.pth"


class PlayerMeta(BaseModel):
    name: str
    is_thinking: bool


class GameStateResponse(BaseModel):
    board: List[List[int]]
    workers: Dict[str, List[int]]
    current_player: str
    players: Dict[str, PlayerMeta]


class StepResponse(GameStateResponse):
    action: Optional[str] = None
    message: Optional[str] = None


class ResetRequest(BaseModel):
    one_type: AgentKind = "greedy"
    two_type: AgentKind = "greedy"


class PlacementMoveRequest(BaseModel):
    type: Literal["placement"]
    to: List[int] = Field(..., min_length=2, max_length=2)
    worker: Optional[str] = None


class TurnMoveRequest(BaseModel):
    type: Literal["turn"]
    worker: str
    from_coord: Optional[List[int]] = Field(default=None, min_length=2, max_length=2)
    to: List[int] = Field(..., min_length=2, max_length=2)
    build: List[int] = Field(..., min_length=2, max_length=2)
    build_dome: bool = False


MoveRequest = Union[PlacementMoveRequest, TurnMoveRequest]


class LegalMovesResponse(BaseModel):
    current_player: str
    moves: List[Dict]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pentgon.com",
        "https://www.pentgon.com",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok"}


game = GameState()

player_types: Dict[Player, AgentKind] = {
    Player.ONE: "greedy",
    Player.TWO: "greedy",
}

agents: Dict[Player, Optional[object]] = {}


def create_agent(player: Player, agent_kind: AgentKind):
    if agent_kind == "human":
        return None
    if agent_kind == "random":
        return RandomAgent(player)
    if agent_kind == "greedy":
        return GreedyAgent(player)
    if agent_kind == "heuristic":
        return HeuristicAgent(player)
    if agent_kind == "minimax-d2":
        return MinimaxAgent(player, depth=2)
    if agent_kind == "vanilla-mcts":
        try:
            from engine.agents.mcts_agent import VanillaMCTSAgent
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail="MCTS agents are unavailable because torch is not installed",
            ) from exc
        return VanillaMCTSAgent(player=player, sims=MCTS_SIMULATIONS)
    if agent_kind == "mcts-nn":
        try:
            from engine.agents.mcts_agent import MCTSAgent
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail="MCTS agents are unavailable because torch is not installed",
            ) from exc
        return MCTSAgent.from_checkpoint(
            player=player,
            checkpoint_path=DEFAULT_CHECKPOINT_PATH,
            sims=MCTS_SIMULATIONS,
            device="auto",
            use_heuristic=False,
        )
    if agent_kind == "mcts-heuristic":
        try:
            from engine.agents.mcts_agent import MCTSAgent
        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail="MCTS agents are unavailable because torch is not installed",
            ) from exc
        return MCTSAgent.from_checkpoint(
            player=player,
            checkpoint_path=DEFAULT_CHECKPOINT_PATH,
            sims=MCTS_SIMULATIONS,
            device="auto",
            use_heuristic=True,
        )
    raise ValueError(f"Unsupported agent type: {agent_kind}")


def refresh_agents() -> None:
    global agents
    agents = {
        Player.ONE: create_agent(Player.ONE, player_types[Player.ONE]),
        Player.TWO: create_agent(Player.TWO, player_types[Player.TWO]),
    }


def worker_label_map() -> Dict[tuple, str]:
    return {
        (Player.ONE, 0): "ONE_A",
        (Player.ONE, 1): "ONE_B",
        (Player.TWO, 0): "TWO_A",
        (Player.TWO, 1): "TWO_B",
    }


def worker_key_map() -> Dict[str, tuple]:
    return {label: key for key, label in worker_label_map().items()}


def coord_to_list(coord) -> List[int]:
    return [coord[0], coord[1]]


def format_action(action: Union[PlacementAction, TurnAction]) -> str:
    if isinstance(action, PlacementAction):
        return f"placement:{worker_label_map()[(action.player, action.worker_index)]}->{coord_to_list(action.coord)}"
    return (
        f"turn:{worker_label_map()[(action.player, action.worker_index)]}:"
        f"{coord_to_list(action.move_from)}->{coord_to_list(action.move_to)}:"
        f"build={coord_to_list(action.build_at)}:dome={action.build_dome}"
    )


def agent_display_name(player: Player) -> str:
    agent_kind = player_types[player]
    if agent_kind == "human":
        return f"Player {player.name.title()}"
    return {
        "random": "Random",
        "greedy": "Greedy",
        "heuristic": "Heuristic",
        "minimax-d2": "Minimax-d2",
        "vanilla-mcts": "Vanilla MCTS",
        "mcts-nn": "MCTS-NN",
        "mcts-heuristic": "MCTS-Heuristic",
    }[agent_kind]


def serialize_workers(state: GameState) -> Dict[str, List[int]]:
    key_map = {
        (Player.ONE, 0): "ONE_A",
        (Player.ONE, 1): "ONE_B",
        (Player.TWO, 0): "TWO_A",
        (Player.TWO, 1): "TWO_B",
    }
    workers: Dict[str, List[int]] = {}

    for worker_key, label in key_map.items():
        coord = state.workers.get(worker_key)
        if coord is not None:
            workers[label] = [coord[0], coord[1]]

    return workers


    


def serialize_game_state(state: GameState) -> GameStateResponse:
    return GameStateResponse(
        board=state.board.heights,
        workers=serialize_workers(state),
        current_player=state.to_move.name,
        players={
            "ONE": PlayerMeta(
                name=agent_display_name(Player.ONE),
                is_thinking=False,
            ),
            "TWO": PlayerMeta(
                name=agent_display_name(Player.TWO),
                is_thinking=False,
            ),
        },
    )


def response_dict(response: BaseModel) -> Dict:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def any_human_players() -> bool:
    return any(kind == "human" for kind in player_types.values())


def apply_action(state: GameState, action: Union[PlacementAction, TurnAction]) -> GameState:
    if isinstance(action, PlacementAction):
        return state.apply_placement(action)
    return state.apply_turn(action)


def auto_advance_ai_turns() -> List[str]:
    global game
    actions_taken: List[str] = []
    if not any_human_players():
        return actions_taken

    while not game.is_over() and player_types[game.to_move] != "human":
        agent = agents.get(game.to_move)
        legal_actions = game.legal_actions()
        if agent is None or not legal_actions:
            break
        action = agent.select_action(game, legal_actions)
        if action is None:
            break
        game = apply_action(game, action)
        actions_taken.append(format_action(action))
    return actions_taken


def action_matches(submitted: Union[PlacementAction, TurnAction], candidate: Union[PlacementAction, TurnAction]) -> bool:
    if type(submitted) is not type(candidate):
        return False
    if isinstance(submitted, PlacementAction):
        return submitted == candidate
    return (
        submitted.player == candidate.player
        and submitted.worker_index == candidate.worker_index
        and submitted.move_from == candidate.move_from
        and submitted.move_to == candidate.move_to
        and submitted.build_at == candidate.build_at
        and submitted.build_dome == candidate.build_dome
    )


def parse_coord(raw: List[int], field_name: str) -> tuple:
    if len(raw) != 2:
        raise HTTPException(status_code=400, detail=f"{field_name} must have exactly 2 integers")
    row, col = raw
    if not (0 <= row < 5 and 0 <= col < 5):
        raise HTTPException(status_code=400, detail=f"{field_name} must be inside the 5x5 board")
    return (row, col)


def parse_move_request(payload: MoveRequest) -> Union[PlacementAction, TurnAction]:
    current_player = game.to_move
    if payload.type == "placement":
        coord = parse_coord(payload.to, "to")
        placed_workers = sum(1 for player, _ in game.workers if player == current_player)
        worker_index = placed_workers
        if payload.worker:
            worker_key = worker_key_map().get(payload.worker)
            if worker_key is None:
                raise HTTPException(status_code=400, detail="Unknown worker id")
            worker_player, requested_index = worker_key
            if worker_player != current_player or requested_index != worker_index:
                raise HTTPException(status_code=400, detail="Invalid worker for current placement turn")
        return PlacementAction(current_player, coord, worker_index)

    worker_key = worker_key_map().get(payload.worker)
    if worker_key is None:
        raise HTTPException(status_code=400, detail="Unknown worker id")

    worker_player, worker_index = worker_key
    if worker_player != current_player:
        raise HTTPException(status_code=400, detail="Submitted worker does not belong to current player")

    move_from = game.worker_coord(worker_player, worker_index)
    if move_from is None:
        raise HTTPException(status_code=400, detail="Worker is not placed on the board")

    if payload.from_coord is not None and tuple(payload.from_coord) != move_from:
        raise HTTPException(status_code=400, detail="from_coord does not match worker position")

    return TurnAction(
        player=current_player,
        worker_index=worker_index,
        move_from=move_from,
        move_to=parse_coord(payload.to, "to"),
        build_at=parse_coord(payload.build, "build"),
        build_dome=payload.build_dome,
    )


def legal_moves_payload() -> List[Dict]:
    moves: List[Dict] = []
    for action in game.legal_actions():
        if isinstance(action, PlacementAction):
            moves.append({
                "type": "placement",
                "worker": worker_label_map()[(action.player, action.worker_index)],
                "to": coord_to_list(action.coord),
            })
        else:
            moves.append({
                "type": "turn",
                "worker": worker_label_map()[(action.player, action.worker_index)],
                "from": coord_to_list(action.move_from),
                "to": coord_to_list(action.move_to),
                "build": coord_to_list(action.build_at),
                "build_dome": action.build_dome,
            })
    return moves


refresh_agents()


@app.get("/game", response_model=GameStateResponse)
def get_game_state() -> GameStateResponse:
    return serialize_game_state(game)


@app.post("/game/reset", response_model=StepResponse)
def reset_game(payload: Optional[ResetRequest] = None) -> StepResponse:
    global game, player_types
    payload = payload or ResetRequest()

    game = GameState()
    player_types = {
        Player.ONE: payload.one_type,
        Player.TWO: payload.two_type,
    }
    refresh_agents()
    actions_taken = auto_advance_ai_turns()

    message = None
    if actions_taken:
        message = " | ".join(actions_taken)

    return StepResponse(
        **response_dict(serialize_game_state(game)),
        message=message,
    )


@app.post("/game/step", response_model=StepResponse)
def game_step() -> StepResponse:
    global game

    if game.is_over():
        return StepResponse(**response_dict(serialize_game_state(game)), message="Game over")

    if player_types[game.to_move] == "human":
        return StepResponse(
            **response_dict(serialize_game_state(game)),
            message="Current player is human; use /game/move",
        )

    agent = agents.get(game.to_move)
    legal_actions = game.legal_actions()
    if not legal_actions or agent is None:
        return StepResponse(
            **response_dict(serialize_game_state(game)),
            message="No actions available",
        )

    action = agent.select_action(game, legal_actions)
    if action is None:
        return StepResponse(
            **response_dict(serialize_game_state(game)),
            message="Agent did not return a valid action",
        )

    game = apply_action(game, action)
    chained_actions = auto_advance_ai_turns()
    action_strings = [format_action(action), *chained_actions]

    return StepResponse(
        **response_dict(serialize_game_state(game)),
        action=" | ".join(action_strings),
    )


@app.post("/game/move", response_model=StepResponse)
def game_move(payload: MoveRequest) -> StepResponse:
    global game

    if game.is_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    if player_types[game.to_move] != "human":
        raise HTTPException(status_code=400, detail="Current player is not human")

    submitted_action = parse_move_request(payload)
    legal_actions = game.legal_actions()
    matched_action = next(
        (action for action in legal_actions if action_matches(submitted_action, action)),
        None,
    )
    if matched_action is None:
        raise HTTPException(status_code=400, detail="Illegal move")

    game = apply_action(game, matched_action)
    chained_actions = auto_advance_ai_turns()
    action_strings = [format_action(matched_action), *chained_actions]

    return StepResponse(
        **response_dict(serialize_game_state(game)),
        action=" | ".join(action_strings),
    )


@app.get("/game/legal_moves", response_model=LegalMovesResponse)
def get_legal_moves() -> LegalMovesResponse:
    return LegalMovesResponse(
        current_player=game.to_move.name,
        moves=legal_moves_payload(),
    )