#!/bin/bash

#SBATCH --job-name=T4-sim-budget-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=tests/T4_sim_budget_sweep_%j.log

echo "T4: Simulation Budget started on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

module load python/3.13.0
module load cuda/12.4.1
source venv/bin/activate

mkdir -p results

for SIMS in 50 100 200 400 800; do
    echo "--- Sims = ${SIMS} at $(date) ---"
    python tournament.py \
        --agents Vanilla-MCTS MCTS-NN MCTS-Heuristic Minimax-d2 \
        --games_per_matchup 50 \
        --mcts_sims ${SIMS} \
        --csv "results/sim_sweep_${SIMS}_summary.csv" \
        --csv_games "results/sim_sweep_${SIMS}_games.csv"
done

echo "T4: Simulation Budget  finished at $(date)"
