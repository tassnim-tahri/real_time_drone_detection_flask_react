# Architecture (production-oriented layout)

## What you run day to day

1. **`detection_pipeline/flask_app.py`** — Primary Python API (**port 8000**): detection, demos, MJPEG streaming, client-camera frames (`/api/client_frame`), **background YOLO video scan** (`/api/analysis/upload` + poll `status`/`report`/`summary`).
2. **`backend/` Express app** (**port 5000**) — Auth, Mongo, optional proxy to Flask via **`FLASK_AI_URL`** (see `STREAMING.md`).

Start both from repo root:

```powershell
npm run dev
```

## Removed prototypes

Standalone duplicates are **deleted** from the tree (duplicate Flask stacks and old `ai_service/`).  
They only exist in **`git history`** — use history or revert if you truly need code from an old demo.

## Hygiene

- Do not commit uploads or generated media; see **`.gitignore`** (`uploads/`, `outputs/`, `analysis_uploads/`, weights patterns).
- Use **environment variables** (`.env` is ignored) for secrets in production (`GROQ_API_KEY`, Mongo URI, etc.).
