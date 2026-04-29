#!/bin/bash

#SBATCH --job-name=bench-nnmcts-greedy
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=bench_nnmcts_vs_greedy_%j.out

echo "Job started on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

module load python/3.13.0
module load cuda/12.4.1
source venv/bin/activate

GAMES=${1:-50}
SIMS=${2:-200}

echo "Benchmark: MCTS-NN vs Greedy  |  Games: $GAMES  |  Sims: $SIMS"

python hpt_tournament.py \
    --model_path checkpoints/koalemos.pth \
    --agents MCTS-NN Greedy \
    --games_per_matchup "$GAMES" \
    --mcts_sims "$SIMS" \
    --csv "bench_nnmcts_vs_greedy.csv"

echo "Finished at $(date)."
