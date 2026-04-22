#!/bin/bash

#SBATCH --job-name=bench-mctsheur-greedy
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=SLURM_mctsheur_vs_greedy_%j.out

cd "$SLURM_SUBMIT_DIR"

module load python/3.13.0
module load cuda/12.4.1
source venv/bin/activate

GAMES=${1:-50}
SIMS=${2:-200}

python tournament.py \
    --model_path checkpoints/koalemos.pth \
    --agents Heuristic-MCTS Greedy \
    --games_per_matchup "$GAMES" \
    --mcts_sims "$SIMS"

echo "Finished at $(date)."
