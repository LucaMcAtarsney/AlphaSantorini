#!/bin/bash

#SBATCH --job-name=T1-full-tournament
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=tests/T1_full_tournament_%j.log

echo "T1: Tournament started on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

module load python/3.13.0
module load cuda/12.4.1
source venv/bin/activate

mkdir -p results

python tournament.py \
    --games_per_matchup 100 \
    --mcts_sims 200 \
    --csv results/full_tournament_summary.csv \
    --csv_games results/full_tournament_games.csv

echo "T1 finished at $(date)"
