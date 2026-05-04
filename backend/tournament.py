from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from itertools import combinations
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

from engine.agents.base import Agent
from engine.agents.greedy_agent import GreedyAgent
from engine.agents.heuristic_agent import HeuristicAgent
from engine.agents.mcts_agent import DEFAULT_CHECKPOINT_PATH, MCTSAgent, VanillaMCTSAgent
from engine.agents.minimax_agent import MinimaxAgent
from engine.agents.random_agent import RandomAgent
from engine.enums import GamePhase, Player
from engine.game_types import PlacementAction, TurnAction
from engine.state import GameState


def format_action(action: PlacementAction | TurnAction) -> str:
    
    # Convert an action into a string for sequence move sequence analysis (might need/ might drop)
    if isinstance(action, PlacementAction):
        row, col = action.coord
        return f"place:w{action.worker_index}@({row},{col})"

    fr, fc = action.move_from
    tr, tc = action.move_to
    br, bc = action.build_at
    dome = 1 if action.build_dome else 0
    return (
        f"turn:w{action.worker_index}:({fr},{fc})->({tr},{tc})"
        f":build({br},{bc}):dome={dome}"
    )


def play_game(
    p1_agent: Agent,
    p2_agent: Agent,
    p1_name: str = "player_one",
    p2_name: str = "player_two",
) -> Tuple[Optional[Player], Dict[str, float], Dict[str, int], float, List[str]]:
    
    # Plays one game - returns winner, timing data, duration, action sequence

    state = GameState()
    agents = {Player.ONE: p1_agent, Player.TWO: p2_agent}
    agent_names = {Player.ONE: p1_name, Player.TWO: p2_name}
    think_times = defaultdict(float)
    decision_counts = defaultdict(int)
    game_start = perf_counter()
    action_sequence: List[str] = []

    while not state.is_over():
        current_player = state.to_move
        agent = agents[current_player]
        actions = state.legal_actions()
        if not actions:
            break

        move_start = perf_counter()
        action = agent.select_action(state.clone(), actions)
        elapsed = perf_counter() - move_start
        think_times[agent_names[current_player]] += elapsed
        decision_counts[agent_names[current_player]] += 1

        if action is None:
            return (
                current_player.opponent(),
                dict(think_times),
                dict(decision_counts),
                perf_counter() - game_start,
                action_sequence,
            )

        action_sequence.append(format_action(action))

        if isinstance(action, PlacementAction):
            state = state.apply_placement(action)
        elif isinstance(action, TurnAction):
            state = state.apply_turn(action)

    return (
        state.winner,
        dict(think_times),
        dict(decision_counts),
        perf_counter() - game_start,
        action_sequence,
    )


def main():
    parser = argparse.ArgumentParser(description=" Santorini round robin tournament")
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint for MCTS agents (default: %(default)s).",
    )
    parser.add_argument(
        "--games_per_matchup", type=int, default=8,
        help="Total games per agent pair (half as each colour).",
    )
    parser.add_argument(
        "--mcts_sims", type=int, default=200,
        help="MCTS simulations per move.",
    )
    parser.add_argument(
        "--agents", type=str, nargs="+", default=None,
        help="Agent names to include (default: all). "
             "Available: Random, Greedy, Heuristic, Minimax-d2, Vanilla-MCTS, MCTS-NN, MCTS-Heuristic",
    )
    parser.add_argument(
        "--csv", type=str, default="tournament_results.csv",
        help="Path to save head2head summary CSV.",
    )
    parser.add_argument(
        "--csv_games", type=str, default=None,
        help="Path to save per game CSV with full timing data.",
    )
    parser.add_argument(
        "--jsonl_moves", type=str, default=None,
        help="Path to save one JSON record per game including the full move sequence.",
    )
    args = parser.parse_args()

    checkpoint = args.model_path
    sims = args.mcts_sims

    all_agent_configs: Dict[str, Callable[[Player], Agent]] = {
        "Random": lambda p: RandomAgent(player=p),
        "Greedy": lambda p: GreedyAgent(player=p),
        "Heuristic": lambda p: HeuristicAgent(player=p),
        "Minimax-d2": lambda p: MinimaxAgent(player=p, depth=2),
        "Vanilla-MCTS": lambda p: VanillaMCTSAgent(player=p, sims=sims),
        "MCTS-NN": lambda p: MCTSAgent.from_checkpoint(
            player=p, checkpoint_path=checkpoint,
            sims=sims, device="auto", use_heuristic=False,
        ),
        "MCTS-Heuristic": lambda p: MCTSAgent.from_checkpoint(
            player=p, checkpoint_path=checkpoint,
            sims=sims, device="auto", use_heuristic=True,
        ),
    }

    if args.agents:
        for name in args.agents:
            if name not in all_agent_configs:
                parser.error(f"Unknown agent '{name}'. Available: {list(all_agent_configs)}")
        agent_configs = {n: all_agent_configs[n] for n in args.agents}
    else:
        agent_configs = all_agent_configs

    agent_names = list(agent_configs.keys())
    matchups = list(combinations(agent_names, 2))

    scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_think_time: Dict[str, float] = defaultdict(float)
    total_decisions: Dict[str, int] = defaultdict(int)
    total_tournament_time = 0.0
    total_completed_games = 0
    game_records: List[Dict] = []

    print(f"\n{'='*60}")
    print(f"Round-Robin Tournament  |  {args.games_per_matchup} games per matchup")
    print(f"Agents: {', '.join(agent_names)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"MCTS sims: {sims}")
    print(f"{'='*60}\n")

    for matchup_idx, (name1, name2) in enumerate(matchups, start=1):
        print(f"[{matchup_idx}/{len(matchups)}] {name1} vs {name2} ... ", end="", flush=True)

        matchup_wins = {name1: 0.0, name2: 0.0}

        half = args.games_per_matchup // 2
        remainder = args.games_per_matchup % 2

        for g in range(args.games_per_matchup):
            # Alternate turn order
            if g % 2 == 0:
                p1_name, p2_name = name1, name2
            else:
                p1_name, p2_name = name2, name1

            p1 = agent_configs[p1_name](Player.ONE)
            p2 = agent_configs[p2_name](Player.TWO)

            winner, think_times, decision_counts, game_time, action_sequence = play_game(
                p1, p2, p1_name, p2_name,
            )
            total_tournament_time += game_time
            total_completed_games += 1

            if winner is None:
                winner_name = "draw"
            elif winner == Player.ONE:
                winner_name = p1_name
            else:
                winner_name = p2_name

            game_records.append({
                "game_id": total_completed_games,
                "agent_a": name1,
                "agent_b": name2,
                "p1_agent": p1_name,
                "p2_agent": p2_name,
                "winner": winner_name,
                "game_duration": round(game_time, 4),
                "think_time_a": round(think_times.get(name1, 0.0), 4),
                "think_time_b": round(think_times.get(name2, 0.0), 4),
                "decisions_a": decision_counts.get(name1, 0),
                "decisions_b": decision_counts.get(name2, 0),
                "move_count": len(action_sequence),
                "mcts_sims": sims,
                "moves": action_sequence,
            })

            for agent_name, elapsed in think_times.items():
                total_think_time[agent_name] += elapsed
            for agent_name, count in decision_counts.items():
                total_decisions[agent_name] += count

            if winner_name == "draw":
                scores[name1][name2] += 0.5
                scores[name2][name1] += 0.5
                matchup_wins[name1] += 0.5
                matchup_wins[name2] += 0.5
            elif winner == Player.ONE:
                scores[p1_name][p2_name] += 1
                matchup_wins[p1_name] += 1
            else:
                scores[p2_name][p1_name] += 1
                matchup_wins[p2_name] += 1

        print(
            f"{name1} {matchup_wins[name1]:.0f}  |  {name2} {matchup_wins[name2]:.0f}"
        )

    total_wins: Dict[str, float] = defaultdict(float)
    total_losses: Dict[str, float] = defaultdict(float)
    total_draws_approx: Dict[str, float] = defaultdict(float)
    total_games: Dict[str, int] = defaultdict(int)

    for a in agent_names:
        for b in agent_names:
            if a == b:
                continue
            total_wins[a] += scores[a][b]
            total_games[a] += args.games_per_matchup

    sorted_agents = sorted(agent_names, key=lambda n: total_wins[n], reverse=True)

    print(f"\n{'='*80}")
    print(
        f"{'Rank':<5} {'Agent':<20} {'Win%':>7} {'Wins':>7} {'Games':>7}"
        f"  {'Decisions':>10} {'Think(s)':>10} {'Avg Move(s)':>12}"
    )
    print("-" * 80)
    for rank, name in enumerate(sorted_agents, start=1):
        games = total_games[name]
        win_pct = 100.0 * total_wins[name] / games if games else 0.0
        decisions = total_decisions[name]
        avg_move = total_think_time[name] / decisions if decisions else 0.0
        print(
            f"{rank:<5} {name:<20} {win_pct:>6.1f}% {total_wins[name]:>7.1f} {games:>7}"
            f"  {decisions:>10} {total_think_time[name]:>10.2f} {avg_move:>12.4f}"
        )
    print("=" * 80)

    avg_game_time = (
        total_tournament_time / total_completed_games if total_completed_games else 0.0
    )
    print(
        f"\nTotal games: {total_completed_games}  |  "
        f"Total time: {total_tournament_time:.1f}s  |  "
        f"Avg game: {avg_game_time:.2f}s"
    )

    # Save Csv
    csv_path = args.csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent_a", "agent_b", "score_a", "score_b", "games"])
        for a, b in matchups:
            w.writerow([a, b, f"{scores[a][b]:.1f}", f"{scores[b][a]:.1f}",
                         args.games_per_matchup])
    print(f"Results saved to {csv_path}")

    # Save per game
    if args.csv_games:
        fieldnames = [
            "game_id", "agent_a", "agent_b", "p1_agent", "p2_agent",
            "winner", "game_duration", "think_time_a", "think_time_b",
            "decisions_a", "decisions_b", "move_count", "mcts_sims",
        ]
        with open(args.csv_games, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for record in game_records:
                w.writerow({key: record[key] for key in fieldnames})
        print(f"Per-game data saved to {args.csv_games}")

    # Save move sequences
    if args.jsonl_moves:
        with open(args.jsonl_moves, "w") as f:
            for record in game_records:
                payload = {
                    **record,
                    "moves": record.get("moves"),
                }
                f.write(json.dumps(payload) + "\n")
        print(f"Move sequences saved to {args.jsonl_moves}")


if __name__ == "__main__":
    main()
