# EPA

EPA has three main parts:

- a Python backend containing the Santorini rules engine, AI agents, training scripts, tournament/evaluation scripts and FastAPI API
- a Godot frontend for interacting with the game visually
- exported desktop builds under `builds/`

The frontend and exported builds do not run the Python game logic by themselves. They call the FastAPI backend over HTTP, so Uvicorn must be running before you can interact with the frontend.

## Project Layout

```text
EPA/
  README.md
  .gitignore
  backend/
    api.py                         # FastAPI game API used by the frontend
    main.py                        # Interactive CLI
    tournament.py                  # Agent tournament runner
    train_selfplay.py              # Self-play training loop
    generate_results_charts.py     # Chart generation from result CSVs
    generate_snapshot_analysis.py  # Snapshot ranking / plot generation
    requirements.txt               # Local backend dependencies
    hpc_requirements.txt           # HPC training dependencies
    engine/                        # Santorini rules, MCTS, agents, neural network
    checkpoints/                   # Model checkpoints
    data/                          # Dataset files
    slurms/                        # HPC SLURM jobs
    tests/                         # Experiment SLURM jobs
    charts/                        # Generated charts
    results/                       # Generated experiment outputs
  frontend/
    project.godot                  # Godot project
    Scenes/                        # Godot scenes
    Scripts/                       # Godot scripts
    export_presets.cfg             # Godot export presets
    *.glb, *.ttf                   # Frontend assets
  builds/
    linux/                         # Linux export
    mac/                           # macOS export
    windows/                       # Windows export
```

Most backend commands assume the current working directory is `backend/`.

```bash
cd backend
```

## Backend Setup

### macOS / Linux

From the repository root:

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

From the repository root:

```powershell
cd backend
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### VS Code Interpreter

Select the Python interpreter inside the backend virtual environment:

```text
backend/venv/bin/python              # macOS / Linux
backend\venv\Scripts\python.exe      # Windows
```

## Run the API First

The Godot frontend is configured to call:

```text
http://127.0.0.1:8000
```

Start the backend API before opening the frontend or exported builds:

```bash
cd backend
source venv/bin/activate
uvicorn api:app --reload
```

On Windows PowerShell:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn api:app --reload
```

Check that the API is running by opening:

```text
http://127.0.0.1:8000/docs
```

Keep this terminal open while using the frontend. If Uvicorn is stopped, the frontend will not be able to fetch the game state, reset games, step agents or submit moves.

## Run the Frontend

### From Godot

1. Start the API with Uvicorn first.
2. Open `frontend/project.godot` in Godot.
3. Run the main scene from the Godot editor.

The frontend talks to the backend through `frontend/Scripts/Network.gd`, which defaults to:

```text
http://127.0.0.1:8000
```

### Exported Builds

Start the API first, then run the build for your platform:

```text
builds/linux/SantoriniLINUX.x86_64
builds/mac/SantoriniMAC.dmg
builds/windows/SantoriniWINDOWS.exe
```

The exported builds also expect the API to be available at `http://127.0.0.1:8000`.

## API Endpoints

The API is defined in `backend/api.py`.

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

The API keeps one in-memory game state. Restarting Uvicorn resets the game.

## Run Backend Checks

### Interactive Helper

```bash
cd backend
source venv/bin/activate
python main.py
```

Choose option `1` for a basic placement/turn test. Option `2` simulates games between selected agents.

### Small Tournament Without Neural Agents

This avoids checkpoints and is the fastest useful tournament check:

```bash
cd backend
source venv/bin/activate
mkdir -p results
python tournament.py \
  --agents Greedy Random \
  --games_per_matchup 4 \
  --csv results/greedy_vs_random_summary.csv \
  --csv_games results/greedy_vs_random_games.csv
```

If this succeeds, `tournament.py`, the game engine and the basic agents are working.

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

The neural/MCTS checkpoint agents expect:

```text
checkpoints/koalemos.pth
```

## HPC Setup

There are two dependency files because the local and HPC environments are different.

```text
requirements.txt      # macOS / Windows / normal Linux local development
hpc_requirements.txt  # HPC dependencies only
```

Use `requirements.txt` on your own machine. Use `hpc_requirements.txt` on the HPC.

The SLURM scripts assume they are submitted from inside `backend/`.

```bash
cd backend
module load python/3.13.0
module load cuda/12.4.1
python -m venv venv
source venv/bin/activate
pip install -r hpc_requirements.txt
```

Do this:

```bash
cd backend
sbatch slurms/tournament.slurm
```

Avoid this:

```bash
sbatch backend/slurms/tournament.slurm
```

Common jobs:

```text
slurms/tournament.slurm                         # Full tournament
slurms/generate_models.slurm                    # Self-play training with snapshots
slurms/eval_snapshots_vs_greedy.slurm           # Evaluate snapshots
slurms/eval_snapshots_vs_greedy_reverse.slurm   # Evaluate snapshots newest first
tests/T1_full_tournament.slurm                  # Full experiment tournament
tests/T2_baseline_agents.slurm                  # Baseline agents
tests/T3_search_agents.slurm                    # Search agents
tests/T4_sim_budget_sweep.slurm                 # Simulation budget sweep
tests/T5_nn_vs_baselines.slurm                  # NN-MCTS vs baselines
```

Snapshot evaluation example:

```bash
cd backend
sbatch slurms/eval_snapshots_vs_greedy_reverse.slurm \
  checkpoints/iterations_koalemos \
  results/snapshot_vs_greedy_reverse \
  100 \
  100
```

The snapshot scripts require `.pth` files in the snapshot directory.

## Training

Training is intended for HPC/GPU use.

```bash
cd backend
sbatch slurms/generate_models.slurm
```

Default training paths:

```text
checkpoints/koalemos.pth
checkpoints/progress_koalemos.json
checkpoints/iterations_koalemos/
```

You can also run the Python training script directly, but this can be slow locally:

```bash
cd backend
source venv/bin/activate
python train_selfplay.py
```

## Common Problems

### Frontend Cannot Connect

Make sure Uvicorn is running before launching the frontend:

```bash
cd backend
source venv/bin/activate
uvicorn api:app --reload
```

Then check:

```text
http://127.0.0.1:8000/docs
```

If the frontend is running on a different machine, update the frontend API base URL or start Uvicorn with `--host 0.0.0.0` on a trusted network.

### `ModuleNotFoundError: No module named 'engine'`

You are probably running from the repository root. Run backend scripts from `backend/`:

```bash
cd backend
python main.py
```

Or set:

```bash
PYTHONPATH=backend
```

### `ModuleNotFoundError: No module named 'torch'`

The active Python environment does not have PyTorch installed or VS Code is using the wrong interpreter. Select the interpreter inside `backend/venv/`, then retry.

### SLURM Cannot Find `venv`, `tournament.py` or `checkpoints`

Submit the job from inside `backend/`:

```bash
cd backend
sbatch tests/T2_baseline_agents.slurm
```
