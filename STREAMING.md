# Streaming plan (architecture)

## What exists now

1. **`flask_app.py` (port 8000)**  
   Unified API: `/detect`, `/analyze_video`, `/chat`, **`/video_feed` (MJPEG)**, **`/api/stats`**, **`/api/set_confidence`**, **`/demo/streaming`**.

2. **`real_time_flask_app.py` (port 8001)**  
   Alternate app with streaming + richer analysis-status APIs.

3. **`live_stream.py`**  
   Shared MJPEG generator and routes (registered inside `flask_app.py`).

## Roadmap — next refinement steps

| Phase | Goal | Technique |
|-------|------|-----------|
| **1 — Done** | One server exposes REST + MJPEG | `register_streaming_routes(app, model)` on port 8000 |
| **2** | One producer, many viewers | Background thread pushes latest JPEG/frame; subscribers read buffered frame (fixes multiple `/video_feed` clients fighting for the camera) |
| **3** | Lower latency metrics | SSE or WebSocket for `/api/stream/events` with JSON payloads (FPS, bbox counts) instead of polling `/api/stats` |
| **4** | Browser WebRTC | For sub-second latency; heavier setup (signaling server, codecs) |
| **5** | Node/BFF proxy | Express proxies `/proxy/video_feed` → Flask so frontend is same origin as JWT cookies |

Env knobs: `CAMERA_INDEX`, `STREAM_CONFIDENCE`, `STREAM_FPS_TARGET`, `YOLO_WEIGHTS_PATH`.

## Run (minimal stack — detect + streaming)

Use a **venv** so Flask 3.x does not clash with other global packages (for example Apache Airflow).

```powershell
cd detection_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python flask_app.py
```

Then open:

- **`http://127.0.0.1:8000/health`** — JSON status
- **`http://127.0.0.1:8000/demo/streaming`** — webcam MJPEG demo (needs a camera)

Optional **full LangChain/VLM pipeline** (`/analyze_video`, `/chat`):

```powershell
pip install -r requirements-llm.txt
```

Heavy imports from `app.py` run only when you call **`/analyze_video`** (lazy load).

**Node backend** (port **5000**): MongoDB must be listening on **`127.0.0.1:27017`**.

```powershell
cd backend
npm install
npm run dev
```

Set **`FLASK_AI_URL=http://127.0.0.1:8000`** when using `POST /api/ai/detect`.

From repo root (both processes; no React app):

```powershell
npm install
npm run dev
```
