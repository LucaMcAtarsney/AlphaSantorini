import os
import pickle
import signal
import sys
import argparse
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp

from santorini.game_types import TurnAction, TOTAL_ACTIONS
from santorini.mcts import MCTS, ActionCodec, encode_state_planes
from santorini.nn.model import PolicyValueNet
from santorini.state import GameState
from santorini.enums import Player


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

shutdown_requested = False

def handle_shutdown(signum, frame):
    global shutdown_requested
    if not shutdown_requested:
        print("\nShutdown requested. Finishing current task and saving progress...")
        shutdown_requested = True

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Promotion
def _play_evaluation_game(
    net_p1: PolicyValueNet,
    net_p2: PolicyValueNet,
    num_simulations: int,
) -> int:
    
    # Play one game, net_p1 as P1 net_p2 as P2
    # Returns +1 if P1 wins, -1 if P2 wins, 0 for draw or interupt

    game = GameState()
    mcts_p1 = MCTS(net_p1, device=DEVICE)
    mcts_p2 = MCTS(net_p2, device=DEVICE)

    while not game.is_over():
        if shutdown_requested:
            return 0

        mcts = mcts_p1 if game.to_move == Player.ONE else mcts_p2
        visit_counts = mcts.run(game, num_simulations=num_simulations)

        best_action = max(visit_counts, key=visit_counts.get)

        if best_action < 50:
            action = ActionCodec.placement_from_index(game.to_move, best_action)
            game = game.apply_placement(action)
        else:
            try:
                action = ActionCodec.decode_action(game, best_action)
                if action in game.legal_actions():
                    game = game.apply_turn(action)
                else:
                    raise ValueError
            except (ValueError, IndexError):
                legal = [a for a in game.legal_actions() if isinstance(a, TurnAction)]
                if legal:
                    game = game.apply_turn(legal[0])
                else:
                    break

    w = game.winner
    if w == Player.ONE:
        return 1
    elif w == Player.TWO:
        return -1
    return 0


def evaluate_candidate(
    champion_net: PolicyValueNet,
    candidate_net: PolicyValueNet,
    num_games: int,
    num_simulations: int,
) -> float:

    # Play num_games between candidate and champion, alternating starting
    # Returns candidate win rate

    candidate_wins = 0
    games_played = 0
    for g in range(num_games):
        if shutdown_requested:
            print("Promotion interrupted by shutdown.")
            break
        if g % 2 == 0:
            # candidate is P1
            result = _play_evaluation_game(candidate_net, champion_net, num_simulations)
            if result == 1:
                candidate_wins += 1
        else:
            # candidate is P2
            result = _play_evaluation_game(champion_net, candidate_net, num_simulations)
            if result == -1:
                candidate_wins += 1
        games_played += 1
        print(f"  Promotion game {games_played}/{num_games}: "
              f"{'candidate won' if (g % 2 == 0 and result == 1) or (g % 2 == 1 and result == -1) else 'champion won or draw'}")
    return candidate_wins / max(games_played, 1)


def self_play(
    net: PolicyValueNet,
    num_simulations: int,
    output_path: str,
) -> str:

    # Plays one game with the MCTS agent against itself
    # Saves training examples to a file and returns path

    game = GameState()
    mcts = MCTS(net, device=DEVICE)

    game_history: List[Tuple[np.ndarray, np.ndarray, Player]] = []

    while not game.is_over():
        if shutdown_requested:
            return ""

        # Get policy for the current state
        visit_counts = mcts.run(game, num_simulations=num_simulations)

        policy_vector = np.zeros(TOTAL_ACTIONS)
        for index, count in visit_counts.items():
            policy_vector[index] = count
        policy_vector /= np.sum(policy_vector)

        # Store state and policy
        state_tensor = encode_state_planes(game)
        game_history.append((state_tensor, policy_vector, game.to_move))

        action_index = np.random.choice(len(policy_vector), p=policy_vector)

        if action_index < 50:
        # Placement action
            action = ActionCodec.placement_from_index(game.to_move, action_index)
            game = game.apply_placement(action)
        else:  
        # Turn action
            found_action = False
            try:
                action = ActionCodec.decode_action(game, action_index)
                if action in game.legal_actions():
                    game = game.apply_turn(action)
                    found_action = True
            except (ValueError, IndexError):
                pass

            if not found_action:
                print(f"Warning: MCTS chose an illegal move index {action_index}. Choosing a random legal move.")
                legal_turn_actions = [act for act in game.legal_actions() if isinstance(act, TurnAction)]
                if legal_turn_actions:
                    game = game.apply_turn(np.random.choice(legal_turn_actions))
                else:
                    print("Error: No legal turn actions available. Ending game.")
                    break

    winner = game.winner
    training_data = []
    for state, policy, player in game_history:
        value = 0
        if winner is not None:
            value = 1 if winner == player else -1
        training_data.append((state, policy, value))

    with open(output_path, "wb") as f:
        pickle.dump(training_data, f)

    return output_path


def save_progress(iteration, net, model_path, progress_path):
    """Saves the current model and iteration number."""
    print("Saving progress...")
    torch.save(net.state_dict(), model_path)
    with open(progress_path, "wb") as f:
        pickle.dump({"iteration": iteration}, f)
    print(f"Progress saved for iteration {iteration}.")


def save_iteration_snapshot(
    net,
    snapshot_dir,
    iteration,
    tag,
    win_rate=None,
    promoted=None,
):
    # Keep an iteration checkpoint so can be tested later
    os.makedirs(snapshot_dir, exist_ok=True)
    checkpoint_path = os.path.join(snapshot_dir, f"iter_{iteration:04d}_{tag}.pth")
    torch.save(net.state_dict(), checkpoint_path)

    metadata = {
        "iteration": iteration,
        "tag": tag,
        "path": checkpoint_path,
    }
    if win_rate is not None:
        metadata["promotion_win_rate"] = win_rate
    if promoted is not None:
        metadata["promoted"] = promoted

    metadata_path = os.path.join(snapshot_dir, f"iter_{iteration:04d}_{tag}.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved snapshot: {checkpoint_path}")

def load_progress(model_path, progress_path):
    # Loads model and  last completed iteration
    net = PolicyValueNet().to(DEVICE)
    start_iter = 0

    # Load weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}...")
        net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("No existing model found. Starting with a new model.")

    # Load training progress if exists
    if os.path.exists(progress_path):
        print(f"Loading training progress from {progress_path}...")
        with open(progress_path, "rb") as f:
            progress = pickle.load(f)
        start_iter = progress.get("iteration", 0)
        print(f"Resuming from iteration {start_iter}.")
    else:
        print("No progress file found. Starting from iteration 0.")

    return start_iter, net


# Main training loop
def train(args):
    start_iter, net = load_progress(args.model_path, args.progress_path)
    print(f"--- Using device: {DEVICE} ---")

    temp_dir = "temp_games"
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(start_iter, args.num_iterations):
        if shutdown_requested:
            save_progress(i, net, args.model_path, args.progress_path)
            print("Exiting.")
            sys.exit(0)

        print(f"\n--- Iteration {i + 1}/{args.num_iterations} ---")

        # 1. Self-play -> generate examples
        print(f"Generating {args.num_games} games using {args.num_workers} workers...")
        all_examples = []
        game_files = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for old_file in glob.glob(os.path.join(temp_dir, "*.pkl")):
                os.remove(old_file)

            futures = [
                executor.submit(
                    self_play,
                    net,
                    args.num_sims,
                    os.path.join(temp_dir, f"game_{i}_{j}.pkl"),
                )
                for j in range(args.num_games)
            ]
            for j, future in enumerate(as_completed(futures)):
                if shutdown_requested:
                    continue
                try:
                    game_file_path = future.result()
                    if game_file_path:
                        game_files.append(game_file_path)
                        print(f"  Game {len(game_files)}/{args.num_games} completed and saved.")
                except Exception as e:
                    print(f"An error occurred during self-play game {j + 1}: {e}")

        if shutdown_requested:
            save_progress(i, net, args.model_path, args.progress_path)
            print("Exiting.")
            sys.exit(0)

        # Load all generated examples from files
        print(f"Loading {len(game_files)} games into memory for training...")
        for game_file in game_files:
            with open(game_file, "rb") as f:
                all_examples.extend(pickle.load(f))
            os.remove(game_file)

        if not all_examples:
            print("No training examples were generated. Skipping training for this iteration.")
            save_progress(i + 1, net, args.model_path, args.progress_path)
            continue

        states, policies, values = zip(*all_examples)
        states_tensor = torch.from_numpy(np.array(states)).float().to(DEVICE)
        policies_tensor = torch.from_numpy(np.array(policies)).float().to(DEVICE)
        values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # 2. Train network
        optimiser = optim.Adam(net.parameters(), lr=args.learning_rate)
        print(f"Training network on {len(all_examples)} examples...")
        for epoch in range(args.num_epochs):
            net.train()
            total_loss = 0
            start_time = time.time()
            for batch_states, batch_policies, batch_values in dataloader:
                optimiser.zero_grad()

                policy_logits, value_pred = net(batch_states)

                # MSE
                value_loss = torch.nn.functional.mse_loss(value_pred, batch_values)

                policy_loss = torch.nn.functional.cross_entropy(policy_logits, batch_policies)

                loss = value_loss + policy_loss
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
            
            epoch_time = time.time() - start_time
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        # 3. Promotion
        if args.no_promotion:
            save_progress(i + 1, net, args.model_path, args.progress_path)
            save_iteration_snapshot(
                net,
                args.snapshot_dir,
                i + 1,
                "champion",
                promoted=True,
            )
            print(f"Iteration {i + 1} complete. Model saved (no promotion gate).")
        else:
            candidate_path = args.model_path + ".candidate"
            torch.save(net.state_dict(), candidate_path)
            save_iteration_snapshot(
                net,
                args.snapshot_dir,
                i + 1,
                "candidate",
            )

            if shutdown_requested:
                with open(args.progress_path, "wb") as f:
                    pickle.dump({"iteration": i + 1}, f)
                if os.path.exists(candidate_path):
                    os.remove(candidate_path)
                print(f"Shutdown during promotion setup. Champion kept. Progress saved at iteration {i + 1}.")
                sys.exit(0)

            # Load champion
            champion_net = PolicyValueNet().to(DEVICE)
            champion_net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
            champion_net.eval()

            candidate_net = PolicyValueNet().to(DEVICE)
            candidate_net.load_state_dict(torch.load(candidate_path, map_location=DEVICE))
            candidate_net.eval()

            print(f"Running promotion: {args.promotion_games} games, "
                  f"threshold {args.promotion_threshold:.0%}...")
            win_rate = evaluate_candidate(
                champion_net, candidate_net,
                num_games=args.promotion_games,
                num_simulations=args.promotion_sims,
            )
            print(f"Candidate win-rate: {win_rate:.1%}")

            # If interrupted mid promotion, reject
            if shutdown_requested:
                net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
                with open(args.progress_path, "wb") as f:
                    pickle.dump({"iteration": i + 1}, f)
                if os.path.exists(candidate_path):
                    os.remove(candidate_path)
                print(f"Shutdown during promotion. Champion kept (partial win-rate {win_rate:.1%}). "
                      f"Progress saved at iteration {i + 1}.")
                sys.exit(0)

            if win_rate >= args.promotion_threshold:
                # Promote - overwrite champion with candidate
                save_progress(i + 1, net, args.model_path, args.progress_path)
                save_iteration_snapshot(
                    net,
                    args.snapshot_dir,
                    i + 1,
                    "champion",
                    win_rate=win_rate,
                    promoted=True,
                )
                print(f"Iteration {i + 1}: PROMOTED (win-rate {win_rate:.1%} >= {args.promotion_threshold:.0%}).")
            else:
                # Reject - reload champion weights back into net for next iteration
                net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
                with open(args.progress_path, "wb") as f:
                    pickle.dump({"iteration": i + 1}, f)
                save_iteration_snapshot(
                    net,
                    args.snapshot_dir,
                    i + 1,
                    "champion_kept",
                    win_rate=win_rate,
                    promoted=False,
                )
                print(f"Iteration {i + 1}: REJECTED (win-rate {win_rate:.1%} < {args.promotion_threshold:.0%}). "
                      f"Keeping champion.")

            if os.path.exists(candidate_path):
                os.remove(candidate_path)

def main():
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Train a model for Santorini through self-play.")
    parser.add_argument("--num_iterations", type=int, default=500, help="Total number of training iterations.")
    parser.add_argument("--num_games", type=int, default=25, help="Number of self-play games to generate per iteration.")
    parser.add_argument("--num_sims", type=int, default=100, help="Number of MCTS simulations per move.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for self-play. Should match CPU cores.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training the neural network.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the network per iteration.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimiser.")
    parser.add_argument("--model_path", type=str, default="checkpoints/pvn.pt", help="Path to load/save the model.")
    parser.add_argument("--progress_path", type=str, default="checkpoints/progress.json", help="Path to load/save training progress.")
    parser.add_argument("--snapshot_dir", type=str, default="checkpoints/iterations", help="Directory to save per-iteration checkpoints for later evaluation.")
    parser.add_argument("--no_promotion", action="store_true", help="Disable promotion gate (always accept new weights).")
    parser.add_argument("--promotion_games", type=int, default=5, help="Number of games for promotion evaluation.")
    parser.add_argument("--promotion_sims", type=int, default=50, help="MCTS sims per move during promotion games (can be lower than training sims).")
    parser.add_argument("--promotion_threshold", type=float, default=0.8, help="Candidate must reach this win-rate to be promoted (0-1).")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
