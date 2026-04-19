from engine.state import GameState
from engine.enums import Player, GamePhase
from engine.game_types import PlacementAction, TurnAction
from engine.agents.random_agent import RandomAgent
from engine.agents.greedy_agent import GreedyAgent
from engine.agents.heuristic_agent import HeuristicAgent
from engine.agents.minimax_agent import MinimaxAgent
from engine.agents.mcts_agent import DEFAULT_CHECKPOINT_PATH, MCTSAgent
from engine.agents.mcts_agent import VanillaMCTSAgent

def print_state(state: GameState):
    print(f"Phase: {state.phase.name}, To Move: {state.to_move.name}")
    if state.winner:
        print(f"Winner: {state.winner.name}")

    worker_map = {
        coord: 'X' if p == Player.ONE else 'Y'
        for (p, _), coord in state.workers.items()
    }

    for r in range(5):
        row_str = ""
        for c in range(5):
            coord = (r, c)
            height = state.board.get_height(coord)
            worker_char = worker_map.get(coord, ' ')
            row_str += f"[{height}{worker_char}]"
        print(row_str)
    print("-" * 20)


def play_single_game(agents, verbose=False):
    game = GameState()
    if verbose:
        print_state(game)

    while not game.is_over():
        legal_actions = game.legal_actions()
        if not legal_actions:
            break

        action = agents[game.to_move].select_action(game.clone(), legal_actions)
        if action is None:
            break

        if isinstance(action, PlacementAction):
            game = game.apply_placement(action)
        elif isinstance(action, TurnAction):
            game = game.apply_turn(action)

        if verbose:
            print(f"Action taken by {game.to_move.opponent().name}: {action}")
            print_state(game)

    return game


def create_agent(choice, player, checkpoint_path):
    if choice == "1":
        return RandomAgent(player=player)
    if choice == "2":
        return GreedyAgent(player=player)
    if choice == "3":
        return HeuristicAgent(player=player)
    if choice == "4":
        return MinimaxAgent(player=player, depth=2)
    if choice == "5":
        return VanillaMCTSAgent(player=player, sims=200)
    if choice == "6":
        return MCTSAgent.from_checkpoint(
            player=player,
            checkpoint_path=checkpoint_path,
            sims=200,
            device="auto",
            use_heuristic=False,
        )
    if choice == "7":
        return MCTSAgent.from_checkpoint(
            player=player,
            checkpoint_path=checkpoint_path,
            sims=200,
            device="auto",
            use_heuristic=True,
        )
    raise ValueError("Invalid agent choice")


def simulate_n_matches():
    agent_map = {
        "1": "Random Agent",
        "2": "Greedy Agent",
        "3": "Heuristic Agent",
        "4": "Minimax Agent",
        "5": "Standard MCTS Agent",
        "6": "NN MCTS Agent",
        "7": "Heuristic MCTS Agent",
    }

    print("\n----------- AGENT SELECTION -----------")
    for key, name in agent_map.items():
        print(f"{key}: {name}")

    checkpoint_path = (
        input(f"Checkpoint path for MCTS agent [{DEFAULT_CHECKPOINT_PATH}]: ").strip()
        or DEFAULT_CHECKPOINT_PATH
    )
    p1_choice = input("Choose player 1 agent (X): ").strip()
    p2_choice = input("Choose player 2 agent (Y): ").strip()
    if p1_choice not in agent_map or p2_choice not in agent_map:
        print("Invalid agent choice")
        return

    try:
        num_games = int(input("Enter number of games: ").strip())
    except ValueError:
        print("Invalid number")
        return

    scores = {Player.ONE: 0, Player.TWO: 0, "draw": 0}
    p1_name = agent_map[p1_choice]
    p2_name = agent_map[p2_choice]
    print(f"\n--- SIMULATING {num_games} GAMES ---")
    print(f"Player 1 (X): {p1_name}")
    print(f"Player 2 (Y): {p2_name}")

    for i in range(num_games):
        agents = {
            Player.ONE: create_agent(p1_choice, Player.ONE, checkpoint_path),
            Player.TWO: create_agent(p2_choice, Player.TWO, checkpoint_path),
        }
        game = play_single_game(agents, verbose=False)

        if game.winner is None:
            scores["draw"] += 1
        else:
            scores[game.winner] += 1

        print(f"Game {i + 1}/{num_games} complete.")

    print(f"\nP1 Wins - {scores[Player.ONE]}")
    print(f"P2 Wins - {scores[Player.TWO]}")
    print(f"Draws - {scores['draw']}")

if __name__ == "__main__":
    print("----------------------- SANTORINI ENGINE -----------------------")
    print("1. Test")
    print("2. Simulate N matches with chosen agents")
    print()
    choice = input(">>> ").strip()
    if choice == '1':
        print("----------------------- TESTING PLACEMENT -----------------------")
        game = GameState()
        print("Initial State:")
        print_state(game)

        game = game.apply_placement(PlacementAction(Player.ONE, (1, 1), 0))
        game = game.apply_placement(PlacementAction(Player.ONE, (1, 3), 1))

        game = game.apply_placement(PlacementAction(Player.TWO, (3, 1), 0))
        game = game.apply_placement(PlacementAction(Player.TWO, (3, 3), 1))

        print("After Placement:")
        print_state(game)

        print("\n----------------------- TESTING TURN -----------------------")
        turn_action = TurnAction(Player.ONE, 0, move_from=(1,1), move_to=(2,2), build_at=(1,2))
        game = game.apply_turn(turn_action)
        print("After Player ONE's first turn:")
        print_state(game)
        
    elif choice == '2':
        simulate_n_matches()
    else:
        print("Invalid choice.")
