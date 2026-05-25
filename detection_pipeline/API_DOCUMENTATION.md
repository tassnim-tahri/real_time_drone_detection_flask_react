# HTTP API — drone detection stack

**Supported server:** run `flask_app.py` (default **port 8000**). All routes below are relative to `http://127.0.0.1:8000` unless noted.

For **how to run**, HTTPS on phones, firewall, and demos, see **`../STREAMING.md`**.

---

## Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, weights path, endpoint list (JSON). |

---

## Detection & analysis

| Method | Path | Description |
|--------|------|-------------|
| POST | `/detect` | Upload **file** → YOLO JSON (existing behavior). |
| POST | `/analyze` | Analysis route (see `flask_app.py` for form fields). |
| POST | `/analyze_video` | Full video pipeline (LangChain/VLM when installed; see `requirements-llm.txt`). |
| POST | `/chat` | Chat against prior report context. |

### Background video scan (YOLO only — poll progress)

Separate from **`/analyze_video`**: frame-by-frame YOLO only, runs in a **background thread**.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analysis/upload` | Multipart **`file`**; optional form **`confidence`** or **`conf`** (0–1, default `0.25`). Saves under **`analysis_uploads/`** and starts the worker. |
| GET | `/api/analysis/status` | `status`: `idle` \| `analyzing` \| `completed` \| `error`, plus frame counts and FPS. |
| GET | `/api/analysis/report` | Full JSON stats + top detection events (**only after** `completed`). |
| GET | `/api/analysis/summary` | Short human-readable summary + stats (**only after** `completed`). |
| POST | `/api/analysis/reset` | Clears analysis state back to **`idle`** (does not cancel an in-flight thread). |


## Live streaming (MJPEG + stats)

Provided by `live_stream.register_streaming_routes`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/video_feed` | MJPEG stream with overlays. |
| POST | `/api/stream/start` | Start/streaming flag helpers. |
| POST | `/api/stream/stop` | Stop/streaming helpers. |
| GET | `/api/stats` | FPS, counts, thresholds (JSON). |
| POST | `/api/set_confidence` | Adjust threshold (JSON body `threshold`). |
| POST | `/api/reset_session` | Reset streamed session counters. |

---

## Browser demos

| Method | Path | Description |
|--------|------|-------------|
| GET | `/demo/streaming` | Server webcam MJPEG page. |
| GET | `/demo/client-camera` | Device camera → `POST /api/client_frame`. |

---

## Device camera inference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/client_frame` | Multipart **`frame`** (JPEG), optional **`conf`**; returns **`detections`** JSON. |

---

## Settings blueprint

Mounted at **`/api/settings/*`** (`settings_routes.py`). Inspect that module for CRUD endpoints.
