from engine.state import GameState
from engine.enums import Player, GamePhase
from engine.game_types import PlacementAction, TurnAction
from engine.agents.random_agent import RandomAgent
from engine.agents.greedy_agent import GreedyAgent
from engine.agents.base import Agent


def print_state(state: GameState):
    print(f"Phase: {state.phase.name}, To Move: {state.to_move.name}")
    if state.winner:
        print(f"Winner: {state.winner.name}")

    # Create a character grid for workers
    worker_map = {
        coord: 'X' if p == Player.ONE else 'Y'
        for (p, _), coord in state.workers.items()
    }

    # Print board with heights and workers
    for r in range(5):
        row_str = ""
        for c in range(5):
            coord = (r, c)
            height = state.board.get_height(coord)
            worker_char = worker_map.get(coord, ' ')
            row_str += f"[{height}{worker_char}]"
        print(row_str)
    print("-" * 20)


def simulate_random_game():
    game = GameState()
    agents = {
        Player.ONE: RandomAgent(),
        Player.TWO: RandomAgent(),
    }

    print("----------------------- SIMULATING RANDOM GAME -----------------------")
    print_state(game)

    while not game.is_over():
        current_player = game.to_move
        agent = agents[current_player]

        legal_actions = game.legal_actions()
        if not legal_actions:
            print(f"No legal actions for {current_player.name}. Game over.")
            break

        action = agent.select_action(game, legal_actions)

        if action is None:
            print(f"Agent for {current_player.name} returned no action. Game over.")
            break

        if isinstance(action, PlacementAction):
            game = game.apply_placement(action)
        elif isinstance(action, TurnAction):
            game = game.apply_turn(action)

        print(f"Action taken by {current_player.name}: {action}")
        print_state(game)

    print("----------------------- GAME OVER -----------------------")
    if game.winner:
        print(f"The winner is {game.winner.name}!")
    else:
        print("The game ended in a draw or was interrupted.")
    print_state(game)


def simulate_n_matches():

    agent_map = {
        "1":("Random Agent", RandomAgent),
        "2":("Greedy Agent", GreedyAgent)
    }

    print("\n----------- AGENT SELECTION -----------")
    for key, (name, _) in agent_map.items():
        print(f"{key}: {name}")

    p1_choice = input("Choose player 1 agent (X): ")
    p2_choice = input("Choose player 2 agent (Y): ")

    if p1_choice not in agent_map or p2_choice not in agent_map:
        print("Invalid agent choice")
        return
    
    p1_agent_class = agent_map[p1_choice][1]
    p2_agent_class = agent_map[p2_choice][1]

    def create_agent(agent_class, player):
        # Suspect some agents will require additional arguments in future
        return agent_class(player=player)

    agents = {
        Player.ONE: create_agent(p1_agent_class, Player.ONE),
        Player.TWO: create_agent(p2_agent_class, Player.TWO)
    }

    p1_name = agent_map[p1_choice][0]
    p2_name = agent_map[p2_choice][0]

    try:
        num_games = int(input("Enter number of games: "))
    except ValueError:
        print("Invalid number")
        return
    
    # --- Simulation Loop ---
    scores = {Player.ONE: 0, Player.TWO: 0, "draw": 0}
    print(f"\n--- SIMULATING {num_games} GAMES ---")
    print(f"Player 1 (X): {p1_name}")
    print(f"Player 2 (Y): {p2_name}")

    print(num_games)

    for i in range(num_games):
        game = GameState()
        while not game.is_over():
            agent = agents[game.to_move]
            legal_actions = game.legal_actions()
            if not legal_actions:
                break
            action = agent.select_action(game, legal_actions)
            if isinstance(action, PlacementAction):
                game = game.apply_placement(action)
            elif isinstance(action, TurnAction):
                game = game.apply_turn(action)
        
        if game.winner:
            scores[game.winner] += 1
        else:
            scores["draw"] += 1
        
        print(f"Game {i + 1}/{num_games} complete...")

    analytics(scores)
    
def analytics(scores : dict):
    
    print(f"P1 Wins - {scores[Player.ONE]}")
    print(f"P2 Wins - {scores[Player.TWO]}")
    print(f"Draws - {scores['draw']}")

if __name__ == "__main__":
    print("----------------------- SANTORINI ENGINE -----------------------")
    print("1. Test Placement")
    print("2. Simulate N matches with chosen agents")
    print()
    choice = input(">>> ")
    if choice == '1':

        # --- Test Placement ---
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

        # --- Test Turn ---
        print("\n----------------------- TESTING TURN -----------------------")
        turn_action = TurnAction(Player.ONE, 0, move_from=(1,1), move_to=(2,2), build_at=(1,2))
        game = game.apply_turn(turn_action)
        print("After Player ONE's first turn:")
        print_state(game)
        
    elif choice == '2':
        simulate_n_matches()
    else:
        print("Invalid choice.")