# EPA

EPA contains a Santorini game engine, AI agents, training/evaluation scripts, and a small FastAPI backend. The project is currently backend-first; a Godot frontend can be added later as a sibling folder at the repository root.

Most backend scripts were developed for an HPC environment with SLURM, CUDA, and a Python virtual environment inside `backend/`. In the current state, commands should be run from the `backend/` directory unless noted otherwise.

## Repository Layout

```text
EPA/
  README.md
  backend/
    api.py                         # FastAPI game API
    main.py                        # Interactive CLI smoke-test/simulation helper
    tournament.py                  # Agent tournament runner
    train_selfplay.py              # Self-play training loop
    generate_results_charts.py     # Chart generation from result CSVs
    generate_snapshot_analysis.py  # Snapshot ranking/plot generation
    requirements.txt               # Cross-platform backend dependencies
    hpc_requirements.txt           # Minimal HPC training dependencies
    engine/                        # Santorini rules, MCTS, agents, neural net
    checkpoints/                   # Model checkpoints
    data/                          # Dataset files
    slurms/                        # HPC SLURM jobs
    tests/                         # Experiment SLURM jobs
    charts/                        # Generated charts
```

## Important Working Directory Note

The backend imports `engine` as a top-level package, for example:

```python
from engine.state import GameState
```

Because of that, run backend commands from inside `backend/`:

```bash
cd backend
```

Running commands from the repository root may fail unless you explicitly set `PYTHONPATH=backend`.

## Environment

The backend has two requirements files:

```text
requirements.txt      # Normal local install for macOS, Windows, or Linux
hpc_requirements.txt  # Minimal HPC training dependencies
```

Typical local setup:

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On HPC, use the minimal HPC requirements file instead. The cluster environment is expected to provide the remaining tools/modules needed by the job scripts:

```bash
module load python/3.13.0
module load cuda/12.4.1
source venv/bin/activate
pip install -r hpc_requirements.txt
```

The checked-in SLURM scripts were written around this HPC setup. The normal `requirements.txt` is intended for local development and includes packages that may not be available on the HPC package indexes.

## Running the API

From `backend/`:

```bash
source venv/bin/activate
uvicorn api:app --reload
```

The API keeps a single in-memory game state. Restarting the server resets that state.

Main endpoints:

```text
GET  /game
POST /game/reset
POST /game/step
POST /game/move
GET  /game/legal_moves
```

Reset example:

```bash
curl -X POST http://127.0.0.1:8000/game/reset \
  -H "Content-Type: application/json" \
  -d '{"one_type":"human","two_type":"greedy"}'
```

Supported API agent types:

```text
human
random
greedy
heuristic
minimax-d2
vanilla-mcts
mcts-nn
mcts-heuristic
```

The MCTS neural-network agents expect a checkpoint at:

```text
backend/checkpoints/koalemos.pth
```

If PyTorch or the checkpoint is unavailable, use non-neural agents such as `human`, `greedy`, `heuristic`, or `minimax-d2`.

## Running a Small CLI Check

There is an interactive helper:

```bash
cd backend
source venv/bin/activate
python main.py
```

This can print a basic placement/turn test or simulate games between selected agents. Neural agents still require PyTorch and a valid checkpoint.

## Running Tournaments

Basic tournament:

```bash
cd backend
source venv/bin/activate
python tournament.py \
  --model_path checkpoints/koalemos.pth \
  --games_per_matchup 10 \
  --mcts_sims 200
```

Restrict to specific agents:

```bash
python tournament.py \
  --agents Random Greedy Heuristic Minimax-d2 \
  --games_per_matchup 100 \
  --csv results/baseline_summary.csv \
  --csv_games results/baseline_games.csv
```

Accepted tournament agent names:

```text
Random
Greedy
Heuristic
Minimax-d2
Vanilla-MCTS
MCTS-NN
MCTS-Heuristic
```

## HPC / SLURM Usage

Submit SLURM jobs from inside `backend/` so relative paths resolve correctly:

```bash
cd backend
sbatch slurms/tournament.slurm
```

Common jobs:

```text
slurms/tournament.slurm                         # Full tournament run
slurms/generate_models.slurm                    # Self-play training with snapshots
slurms/eval_snapshots_vs_greedy.slurm           # Evaluate snapshots oldest/default order
slurms/eval_snapshots_vs_greedy_reverse.slurm   # Evaluate snapshots newest first
tests/T1_full_tournament.slurm                  # Full experiment tournament
tests/T2_baseline_agents.slurm                  # Baseline agent experiment
tests/T3_search_agents.slurm                    # Search-agent experiment
tests/T4_sim_budget_sweep.slurm                 # Simulation budget sweep
tests/T5_nn_vs_baselines.slurm                  # NN-MCTS vs baseline agents
```

Example with arguments:

```bash
sbatch slurms/eval_snapshots_vs_greedy_reverse.slurm \
  checkpoints/iterations_koalemos \
  results/snapshot_vs_greedy_reverse \
  100 \
  100
```

The SLURM scripts assume:

```text
backend/venv/
backend/checkpoints/
backend/results/
backend/tournament.py
backend/train_selfplay.py
```

## Training

Self-play training is handled by:

```bash
python train_selfplay.py
```

The HPC-oriented training script is usually preferred:

```bash
cd backend
sbatch slurms/generate_models.slurm
```

Default paths used by the training scripts:

```text
checkpoints/koalemos.pth
checkpoints/progress_koalemos.json
checkpoints/iterations_koalemos/
```

Training is computationally expensive and was designed for GPU-backed HPC runs.

## Charts and Snapshot Analysis

After tournament/test jobs produce CSV files under `backend/results/`, generate charts with:

```bash
cd backend
source venv/bin/activate
python generate_results_charts.py
```

Snapshot analysis example:

```bash
python generate_snapshot_analysis.py \
  results/snapshot_vs_greedy_reverse \
  --csv-out results/snapshot_ranked.csv \
  --plot-out charts/snapshot_vs_greedy_reverse_progress.svg \
  --title "Checkpoint win rate vs Greedy over training"
```

These scripts expect the relevant CSV outputs to already exist.

## Current Limitations

- The backend is not packaged as an installable Python package yet.
- Most paths are relative to `backend/`.
- Training and large evaluations assume HPC access.
- Local testing depends on having compatible PyTorch installed.
- The API stores game state in memory and is intended for a single active game/session.
