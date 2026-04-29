#!/bin/bash

#SBATCH --job-name=T5-nn-vs-baselines
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=tests/T5_nn_vs_baselines_%j.log

echo "T5: MCTS-NN vs Baselines started on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

module load python/3.13.0
module load cuda/12.4.1
source venv/bin/activate

mkdir -p results

for OPPONENT in Random Greedy Heuristic Minimax-d2; do
    echo "--- MCTS-NN vs ${OPPONENT} at $(date) ---"
    python tournament.py \
        --agents MCTS-NN ${OPPONENT} \
        --games_per_matchup 100 \
        --mcts_sims 200 \
        --csv "results/nn_vs_${OPPONENT}_summary.csv" \
        --csv_games "results/nn_vs_${OPPONENT}_games.csv"
done

echo "T5: MCTS-NN vs Baselines finished at $(date)"
