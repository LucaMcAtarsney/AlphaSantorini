#!/bin/bash

#SBATCH --job-name=santorini-hpt
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=SLURM_hpt_tournament_%j.out

cd "$SLURM_SUBMIT_DIR"

echo "Loading modules..."
module load python/3.13.0
module load cuda/12.4.1

source venv/bin/activate

MODEL_PATH=${1:-checkpoints/koalemos.pth}
GAMES=${2:-10}
SIMS=${3:-200}

python tournament.py \
    --model_path "$MODEL_PATH" \
    --games_per_matchup "$GAMES" \
    --mcts_sims "$SIMS"

echo "Tournament finished at $(date)."