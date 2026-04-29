#!/bin/bash

#SBATCH --job-name=santorini-train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=selfplay_train_%j.out

set -e

echo "Job started on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

echo "Loading modules..."
module load python/3.13.0
module load cuda/12.4.1

echo "Activating virtual environment..."
source venv/bin/activate

MODEL_PATH=${1:-checkpoints/koalemos.pth}
PROGRESS_PATH=${2:-checkpoints/progress_koalemos.json}
SNAPSHOT_DIR=${3:-checkpoints/iterations_koalemos}

mkdir -p "$(dirname "$MODEL_PATH")"
mkdir -p "$(dirname "$PROGRESS_PATH")"
mkdir -p "$SNAPSHOT_DIR"

echo "Model: $MODEL_PATH"
echo "Progress: $PROGRESS_PATH"
echo "Snapshots: $SNAPSHOT_DIR"
echo "Starting self-play training with promotion..."

python train_selfplay.py \
    --model_path "$MODEL_PATH" \
    --progress_path "$PROGRESS_PATH" \
    --snapshot_dir "$SNAPSHOT_DIR" \
    --num_iterations 1000 \
    --num_games 25 \
    --num_sims 100 \
    --num_workers 8 \
    --batch_size 64 \
    --num_epochs 10 \
    --learning_rate 0.001 \
    --promotion_games 5 \
    --promotion_sims 50 \
    --promotion_threshold 1.0

echo "Job finished at $(date)."
