#!/bin/bash

#SBATCH --job-name=T2-baseline-agents
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=tests/T2_baseline_agents_%j.log

echo "T2: Baseline Agents started on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

module load python/3.13.0
module load cuda/12.4.1
source venv/bin/activate

mkdir -p results

python tournament.py \
    --agents Random Greedy Heuristic Minimax-d2 \
    --games_per_matchup 100 \
    --mcts_sims 200 \
    --csv results/baseline_summary.csv \
    --csv_games results/baseline_games.csv

echo "T2 finished at $(date)"
