# Deploying the AlphaSantorini backend to Render

The FastAPI backend (`backend/`, app `api:app`) runs the Santorini engine and AI
agents. The Godot web build is a thin HTTP client that calls it.

## Render settings

Create a new **Web Service** from this repo. Either let Render read `render.yaml`
at the repo root (Blueprint), or set these manually:

| Setting           | Value                                             |
| ----------------- | ------------------------------------------------- |
| Root Directory    | `backend`                                         |
| Runtime           | Python 3                                           |
| Build Command     | `pip install -r requirements.txt`                 |
| Start Command     | `uvicorn api:app --host 0.0.0.0 --port $PORT`     |
| Health Check Path | `/`                                               |
| Instance Type     | **Paid** (Starter or higher — CPU PyTorch needs the RAM) |

`requirements.txt` installs CPU-only PyTorch via
`--extra-index-url https://download.pytorch.org/whl/cpu`, so no GPU is required.

After the first deploy, the service URL looks like
`https://<name>.onrender.com`. Verify:

- `GET /`            → `{"status": "ok"}`  (also the health check / wake endpoint)
- `GET /game`        → current board state
- `POST /game/reset` → resets the game
- `POST /game/step`  → advances one AI move

## Frontend wiring

The web build reads its API base URL from the `api` query-string parameter, e.g.
`index.html?api=https://<name>.onrender.com`. No re-export is needed if the URL
changes — just update the `?api=` value in the embedding iframe. With no param it
falls back to `http://127.0.0.1:8000` for local/desktop development.

## Notes

- CORS is configured in `api.py` for `https://pentgon.com`,
  `https://www.pentgon.com`, and `http://localhost:3000`.
- `checkpoints/koalemos.pth` (~1.5 MB) is committed to git and loaded at runtime
  by the neural-net MCTS agent from the relative path `checkpoints/koalemos.pth`
  (working dir = `backend`).
- The full HPC/analysis dependency set (matplotlib, pandas, seaborn, torchvision,
  torchaudio) lives in `hpc_requirements.txt` and is intentionally excluded from
  the lean runtime `requirements.txt`.
