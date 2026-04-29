#!/bin/bash

#SBATCH --job-name=T3-search-agents
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=tests/T3_search_agents_%j.log

echo "T3: Search Agents started on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

module load python/3.13.0
module load cuda/12.4.1
source venv/bin/activate

mkdir -p results

python tournament.py \
    --agents Minimax-d2 Vanilla-MCTS MCTS-NN MCTS-Heuristic \
    --games_per_matchup 100 \
    --mcts_sims 200 \
    --csv results/search_agents_summary.csv \
    --csv_games results/search_agents_games.csv

echo "T3 finished at $(date)"
