# Streaming plan (architecture)

## What exists now

1. **`flask_app.py` (port 8000)**  
   Unified API: `/detect`, `/analyze_video`, `/chat`, **`/video_feed` (MJPEG)**, **`/api/stats`**, **`/api/set_confidence`**, **`/demo/streaming`**.  
   **Device camera:** **`/demo/client-camera`** captures with `getUserMedia` on whoever opened the page (phone ↔ phone camera); frames are **`POST`**ed as JPEG to **`/api/client_frame`** for YOLO (not MJPEG/WebRTC).

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

Env knobs: `CAMERA_INDEX`, `STREAM_CONFIDENCE`, `STREAM_FPS_TARGET`, `YOLO_WEIGHTS_PATH`, **`CLIENT_FRAME_IMGSZ`** (default `416` — smaller speeds phone→server `/api/client_frame` on CPU laptops). **Mobile client camera (`/demo/client-camera`):** use `FLASK_DEV_HTTPS=1` then open **`https://&lt;lan-ip&gt;:8000`** (see below).

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
- **`http://127.0.0.1:8000/demo/streaming`** — server webcam MJPEG (camera on Flask host).
- **`/demo/client-camera` on laptop:** `http://127.0.0.1:8000/demo/client-camera` (localhost allows camera in most browsers).
- **`/demo/client-camera` on phone over LAN:** browsers block camera on plain `http://192.168.x.x`. Enable dev HTTPS, then open `https://<laptop-ip>:8000/demo/client-camera` and trust the certificate:

```powershell
pip install cryptography
$env:FLASK_DEV_HTTPS = "1"
python flask_app.py
```

**Chrome / Safari “cannot prove that it is 192.168.x.x / attacker”**: that is **normal** with `adhoc` TLS. The cert is **self-signed**, so browsers show a scary message even when the server really is your PC. Only **continue past the warning on your trusted home/office Wi‑Fi** (never for random public sites).

- **Chrome (Android):** try **Advanced** → **Proceed to … (unsafe)** (wording varies).
- **Chrome (desktop):** **Advanced** → **Proceed to …**
- To **remove** the warning: install **[mkcert](https://github.com/FiloSottile/mkcert)** on your PC, run `mkcert -install`, then generate a cert **including your LAN IP**, e.g. `mkcert 192.168.1.18 127.0.0.1 localhost`. Point Flask at those files (**you must trust mkcert’s CA on the phone** too):

```powershell
$env:FLASK_DEV_HTTPS = "1"
$env:FLASK_SSL_CERT = "C:\path\to\<name>.pem"
$env:FLASK_SSL_KEY = "C:\path\to\<name>-key.pem"
python flask_app.py
```

(On Android you typically install mkcert’s **root CA** under Settings → Security → Install certificate, after exporting `mkcert -CAROOT` from the PC.)

**Phone shows “can't be reached / timed out” while laptop works?** Windows often blocks inbound LAN traffic on port 8000. Run **`detection_pipeline/scripts/allow_firewall_port.ps1` as Administrator** once. Confirm both devices use the **same Wi‑Fi** (not isolated guest LAN).

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
