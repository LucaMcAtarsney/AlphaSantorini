#!/bin/bash

#SBATCH --job-name=nnmcts-vs-greedy-rev
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=eval_snapshots_vs_greedy_reverse_%j.out

set -e

echo "Job started on $(hostname) at $(date)"
cd "$SLURM_SUBMIT_DIR"

echo "Loading modules..."
module load python/3.13.0
module load cuda/12.4.1

echo "Activating virtual environment..."
source venv/bin/activate

SNAPSHOT_DIR=${1:-checkpoints/iterations_koalemos}
RESULTS_DIR=${2:-results/snapshot_vs_greedy_reverse}
GAMES_PER_MATCHUP=${3:-100}
MCTS_SIMS=${4:-100}

mkdir -p "$RESULTS_DIR"

echo "Snapshot dir: $SNAPSHOT_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Games per matchup: $GAMES_PER_MATCHUP"
echo "MCTS sims: $MCTS_SIMS"

mapfile -t SNAPSHOTS < <(find "$SNAPSHOT_DIR" -maxdepth 1 -name "*.pth" | sort -r)

if [ ${#SNAPSHOTS[@]} -eq 0 ]; then
    echo "No snapshot .pth files found in $SNAPSHOT_DIR"
    exit 1
fi

echo "Found ${#SNAPSHOTS[@]} snapshots."
echo "Starting from latest snapshot: ${SNAPSHOTS[0]}"

for SNAPSHOT in "${SNAPSHOTS[@]}"; do
    BASENAME=$(basename "$SNAPSHOT" .pth)

    echo "Evaluating $BASENAME vs Greedy at $(date)"
    echo "Checkpoint: $SNAPSHOT"

    python tournament.py \
        --model_path "$SNAPSHOT" \
        --agents MCTS-NN Greedy \
        --games_per_matchup "$GAMES_PER_MATCHUP" \
        --mcts_sims "$MCTS_SIMS" \
        --csv "$RESULTS_DIR/${BASENAME}_summary.csv" \
        --csv_games "$RESULTS_DIR/${BASENAME}_games.csv"
done

echo "Snapshot vs Greedy reverse evaluation finished at $(date)"
