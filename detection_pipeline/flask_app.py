from pathlib import Path
import os
import re
import threading
import time
import traceback
import uuid
from collections import Counter
from typing import Any

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# =============================
# INIT APP
# =============================
from settings_routes import settings_bp
from yolo_weights import resolve_yolo_weights
from live_stream import register_streaming_routes

app = Flask(__name__)
app.register_blueprint(settings_bp)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model once (use YOLO_WEIGHTS_PATH or model.pt / weights — see yolo_weights.py)
_WEIGHTS = resolve_yolo_weights()
print(f"[flask_app] Loading YOLO weights: {_WEIGHTS}")
model = YOLO(_WEIGHTS)
register_streaming_routes(app, model)

# --- Background video scan (YOLO only) for /api/analysis/* progress API ---
ANALYSIS_UPLOAD_DIR = Path("analysis_uploads")
ANALYSIS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AnalysisState:
    """Progress + results for POST /api/analysis/upload (background YOLO scan)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle, analyzing, completed, error
        self.progress = 0
        self.current_video = None
        self.total_frames = 0
        self.processed_frames = 0
        self.fps = 0.0
        self.total_detections = 0
        self.frames_with_detections = 0
        self.class_counts = {}
        self.timeline_data = {}
        self.top_events = []
        self.tracking_data = {
            "enabled": True,
            "unique_tracks": 0,
            "top_tracks": [],
        }
        self.error_message = None


analysis_state = AnalysisState()


def _analysis_get_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _analysis_normalize_bbox(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(xyxy[0]))))
    y1 = max(0, min(height - 1, int(round(xyxy[1]))))
    x2 = max(0, min(width, int(round(xyxy[2]))))
    y2 = max(0, min(height, int(round(xyxy[3]))))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def analyze_video_real_time(video_path: str, conf_threshold: float = 0.25) -> None:
    """Background YOLO frame scan; updates ``analysis_state`` for polling clients."""
    try:
        with analysis_state.lock:
            analysis_state.status = "analyzing"
            analysis_state.progress = 0
            analysis_state.current_video = video_path
            analysis_state.error_message = None
            analysis_state.top_events = []

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid video dimensions")

        class_counts = Counter()
        timeline_counts = Counter()
        top_events = []
        frame_index = 0
        frames_with_detection = 0

        with analysis_state.lock:
            analysis_state.total_frames = total_frames
            analysis_state.fps = fps

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            prediction = model.predict(
                source=frame,
                conf=conf_threshold,
                device=None,
                verbose=False,
            )
            result = prediction[0]
            detections_this_frame = 0
            boxes = result.boxes

            if boxes is not None and boxes.cls is not None and boxes.conf is not None and boxes.xyxy is not None:
                class_ids = boxes.cls.tolist()
                confidences = boxes.conf.tolist()
                xyxy_values = boxes.xyxy.tolist()

                for class_id_float, confidence, xyxy in zip(class_ids, confidences, xyxy_values):
                    class_id = int(class_id_float)
                    label = _analysis_get_class_name(model.names, class_id)
                    yolo_conf = round(float(confidence), 4)

                    class_counts[label] += 1
                    detections_this_frame += 1

                    second_mark = int(frame_index / fps) if fps > 0 else 0
                    timeline_counts[second_mark] += 1

                    x1, y1, x2, y2 = _analysis_normalize_bbox(xyxy, width, height)
                    top_events.append(
                        {
                            "frame": frame_index,
                            "time_sec": round(frame_index / fps, 2) if fps > 0 else 0.0,
                            "label": label,
                            "confidence": yolo_conf,
                            "bbox_xyxy": [x1, y1, x2, y2],
                        }
                    )

            if detections_this_frame > 0:
                frames_with_detection += 1

            frame_index += 1

            with analysis_state.lock:
                analysis_state.processed_frames = frame_index
                analysis_state.progress = (
                    int((frame_index / total_frames) * 100) if total_frames > 0 else 0
                )
                analysis_state.total_detections = int(sum(class_counts.values()))
                analysis_state.frames_with_detections = frames_with_detection
                analysis_state.class_counts = dict(class_counts)
                analysis_state.timeline_data = {
                    str(k): v for k, v in sorted(timeline_counts.items())
                }

        capture.release()

        top_events.sort(key=lambda x: x["confidence"], reverse=True)
        top_events = top_events[:50]

        with analysis_state.lock:
            analysis_state.top_events = top_events
            analysis_state.status = "completed"
            analysis_state.progress = 100

    except Exception as e:
        traceback.print_exc()
        with analysis_state.lock:
            analysis_state.status = "error"
            analysis_state.error_message = str(e)

chain = None
reportText=None

def save_file(file):
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path

@app.route("/health", methods=["GET"])
def health():
    with analysis_state.lock:
        analysis_snapshot = analysis_state.status
    return jsonify(
        {
            "status": "ok",
            "service": "AI + streaming",
            "weights": _WEIGHTS,
            "analysis_progress_api": analysis_snapshot,
            "endpoints": {
                "detect": "/detect",
                "video_feed": "/video_feed",
                "stats": "/api/stats",
                "streaming_demo": "/demo/streaming",
                "client_camera_demo": "/demo/client-camera",
                "client_frame": "POST /api/client_frame",
                "analyze_video": "POST /analyze_video",
                "analysis_upload": "POST /api/analysis/upload",
                "analysis_status": "GET /api/analysis/status",
            },
        }
    )


@app.route("/demo/streaming")
def streaming_demo_page():
    static_dir = Path(__file__).resolve().parent / "static"
    return send_from_directory(static_dir, "streaming_demo.html")


@app.route("/demo/client-camera")
def client_camera_demo_page():
    """Browser uses getUserMedia — camera is on whatever device opened the page (phone or laptop)."""
    static_dir = Path(__file__).resolve().parent / "static"
    return send_from_directory(static_dir, "client_camera_demo.html")


@app.route("/api/client_frame", methods=["POST"])
def client_frame():
    """
    Run YOLO on one JPEG frame uploaded from the visitor's browser camera.
    multipart form: field 'frame' (image/jpeg), optional 'conf' (0-1).
    """
    if "frame" not in request.files:
        return jsonify({"success": False, "error": "No frame uploaded"}), 400
    try:
        conf = float(request.form.get("conf", os.environ.get("STREAM_CONFIDENCE", "0.35")))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = float(os.environ.get("STREAM_CONFIDENCE", "0.35"))

    blob = request.files["frame"].read()
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Could not decode image"}), 400

    try:
        # Smaller imgsz dramatically speeds laptop-CPU inference for phone uploads.
        _imgsz = int(os.environ.get("CLIENT_FRAME_IMGSZ", "416"))
        _imgsz = max(128, min(1280, _imgsz))

        results = model(img, verbose=False, conf=conf, imgsz=_imgsz)
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": name,
                        "confidence": float(box.conf[0]),
                        "bbox": {
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                        },
                    }
                )

        return jsonify(
            {
                "success": True,
                "detections": detections,
                "count": len(detections),
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = save_file(file)

    try:
        results = model(filepath)

        detections = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "label": name,
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                })

        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        return jsonify({
            "error": "Detection failed",
            "details": str(e)
        }), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Placeholder for:
    - Drone threat analysis (VLM)
    - Report generation (LLM)
    """

    data = request.json

    # Example structure expected from frontend later
    detections = data.get("detections", [])

    # Dummy response (replace later)
    return jsonify({
        "summary": "No threat detected",
        "risk_level": "low",
        "details": f"{len(detections)} objects analyzed"
    })

OUTPUT_REPORT_PATH = "outputs/final_threat_report.md"


def parse_report(report_text):
    data = {}

    # ------------------ QUICK SUMMARY ------------------
    qs = {}
    patterns = {
        "drones_detected": r"Drones detected:\s*(\d+)",
        "time_monitored": r"Time monitored:\s*([\d.]+)",
        "frames_with_drones": r"Frames containing drones:\s*(\d+)",
        "unique_drones": r"Unique drones tracked:\s*(\d+)",
        "risk_level": r"Overall risk level:\s*(\w+)",
        "highest_risk": r"Highest risk category observed:\s*(\w+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, report_text)
        if match:
            qs[key] = match.group(1)

    data["quick_summary"] = qs

    # ------------------ MAIN FINDINGS ------------------
    mf = {}
    patterns2 = {
        "drone_type": r"Most common drone type:\s*(.+)",
        "payload": r"Most common payload type:\s*(.+)",
        "size": r"Common drone size:\s*(.+)",
        "threat": r"Threat distribution:\s*(.+)",
    }

    for key, pattern in patterns2.items():
        match = re.search(pattern, report_text)
        if match:
            mf[key] = match.group(1)

    data["main_findings"] = mf

    # ------------------ ACTIONS ------------------
    actions = re.findall(r"- (.+)", report_text)
    data["actions"] = actions

    # ------------------ SUMMARY TEXT ------------------
    summary_match = re.search(r"Plain-Language Summary\n\n(.+)", report_text, re.DOTALL)
    if summary_match:
        data["summary"] = summary_match.group(1).strip()

    return data


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    print(request.form)
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    if "conf" not in request.form:
        return jsonify({"success": False, "error": "No conf sent"}), 400
    if "top_K" not in request.form:
        return jsonify({"success": False, "error": "No top_K sent"}), 400
    if "llmProvider" not in request.form:
        return jsonify({"success": False, "error": "No LLM provider sent"}), 400

    video = request.files["file"]
    conf = float(request.form["conf"])
    top_K = int(request.form["top_K"])
    llmProvider = str(request.form["llmProvider"])
    filename = secure_filename(video.filename)
    input_path = os.path.join("uploads", filename)
    video.save(input_path)

    try:
        # Heavy LangChain / VLM deps — only load when analysis is requested.
        from app import run_full_pipeline

        global chain
        global reportText
        result,chain = run_full_pipeline(input_path,conf,top_K,llmProvider)
        reportText = result["raw_report"]
        
        return jsonify({
            "success": True,
            "report_path": result["report_path"],
            "report_text": result["report_text"],
            "report_from": result["report_from"],
            "report": result["raw_report"],
            "summary": result["summary"]
        })
    except Exception as e:
        print("Pipeline error:", str(e))
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Pipeline failed: {str(e)}"}), 500


@app.route("/api/analysis/upload", methods=["POST"])
def upload_video_for_analysis():
    """Upload a video and start background YOLO scan (poll ``/api/analysis/status``)."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400

    with analysis_state.lock:
        if analysis_state.status == "analyzing":
            return jsonify(
                {
                    "success": False,
                    "error": "Analysis already in progress — wait or POST /api/analysis/reset",
                }
            ), 409

    try:
        raw_conf = request.form.get("confidence") or request.form.get("conf") or "0.25"
        conf_threshold = max(0.0, min(1.0, float(raw_conf)))
    except (TypeError, ValueError):
        conf_threshold = 0.25

    try:
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = ANALYSIS_UPLOAD_DIR / filename
        file.save(str(filepath))

        thread = threading.Thread(
            target=analyze_video_real_time,
            args=(str(filepath.resolve()), conf_threshold),
            daemon=True,
        )
        thread.start()

        return jsonify(
            {
                "success": True,
                "message": "Analysis started",
                "video_file": filename,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analysis/status", methods=["GET"])
def get_analysis_status():
    with analysis_state.lock:
        return jsonify(
            {
                "status": analysis_state.status,
                "progress": analysis_state.progress,
                "current_video": analysis_state.current_video,
                "processed_frames": analysis_state.processed_frames,
                "total_frames": analysis_state.total_frames,
                "total_detections": analysis_state.total_detections,
                "frames_with_detections": analysis_state.frames_with_detections,
                "fps": analysis_state.fps,
                "error_message": analysis_state.error_message,
            }
        )


@app.route("/api/analysis/report", methods=["GET"])
def get_analysis_report():
    with analysis_state.lock:
        if analysis_state.status != "completed":
            return jsonify(
                {"success": False, "error": "Analysis not completed"}
            ), 400
        ratio = (
            round(
                analysis_state.frames_with_detections / analysis_state.total_frames,
                4,
            )
            if analysis_state.total_frames > 0
            else 0
        )
        return jsonify(
            {
                "success": True,
                "status": analysis_state.status,
                "fps": analysis_state.fps,
                "total_frames": analysis_state.total_frames,
                "total_detections": analysis_state.total_detections,
                "frames_with_detections": analysis_state.frames_with_detections,
                "detection_frame_ratio": ratio,
                "class_counts": analysis_state.class_counts,
                "timeline_detections_per_second": analysis_state.timeline_data,
                "top_confidence_events": analysis_state.top_events[:50],
                "tracking": analysis_state.tracking_data,
            }
        )


@app.route("/api/analysis/summary", methods=["GET"])
def get_analysis_summary():
    with analysis_state.lock:
        if analysis_state.status != "completed":
            return jsonify(
                {"success": False, "error": "Analysis not completed"}
            ), 400

        total_frames = analysis_state.total_frames
        duration_sec = (
            total_frames / analysis_state.fps if analysis_state.fps > 0 else 0
        )
        total_detections = analysis_state.total_detections
        frames_with_detections = analysis_state.frames_with_detections
        class_counts = analysis_state.class_counts or {}

        if not class_counts:
            summary = (
                f"Processed {total_frames} frames ({duration_sec:.1f}s). "
                "No detections found."
            )
            top_cls = None
        else:
            top_class, top_count = sorted(
                class_counts.items(), key=lambda x: x[1], reverse=True
            )[0]
            ratio_pct = (
                (frames_with_detections / total_frames * 100)
                if total_frames > 0
                else 0
            )
            summary = (
                f"Processed {total_frames} frames ({duration_sec:.1f}s), found "
                f"{total_detections} detections in {frames_with_detections} frames "
                f"({ratio_pct:.1f}%). Most frequent class: {top_class} ({top_count})."
            )
            top_cls = top_class

        return jsonify(
            {
                "success": True,
                "summary": summary,
                "stats": {
                    "total_frames": total_frames,
                    "duration_sec": round(duration_sec, 2),
                    "total_detections": total_detections,
                    "frames_with_detections": frames_with_detections,
                    "top_class": top_cls,
                },
            }
        )


@app.route("/api/analysis/reset", methods=["POST"])
def reset_analysis_state():
    with analysis_state.lock:
        analysis_state.status = "idle"
        analysis_state.progress = 0
        analysis_state.current_video = None
        analysis_state.total_frames = 0
        analysis_state.processed_frames = 0
        analysis_state.fps = 0.0
        analysis_state.total_detections = 0
        analysis_state.frames_with_detections = 0
        analysis_state.class_counts = {}
        analysis_state.timeline_data = {}
        analysis_state.top_events = []
        analysis_state.error_message = None
    return jsonify({"success": True, "message": "Analysis state reset"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        
        global chain , reportText
        print(chain)
        try : 
            response = chain.invoke({
                "report": reportText,"question": user_message
            })
            return jsonify({
                "response": response
            })
        except Exception as exc:
            print(f"Assistant: Could not answer with LangChain provider: {exc}\n")
            return jsonify({"response" : f"Assistant: Could not answer with LangChain provider: {exc}\n"})

    except Exception as e:
        print("ERROR",str(e))
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    _port = int(os.environ.get("PORT", "8000"))
    _dev_https = os.environ.get("FLASK_DEV_HTTPS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    _ssl_cert = os.environ.get("FLASK_SSL_CERT", "").strip()
    _ssl_key = os.environ.get("FLASK_SSL_KEY", "").strip()
    ssl_context = None
    if _dev_https:
        if _ssl_cert and _ssl_key:
            ssl_context = (_ssl_cert, _ssl_key)
        elif _ssl_cert or _ssl_key:
            print(
                "[flask_app] TLS: set both FLASK_SSL_CERT and FLASK_SSL_KEY "
                "(or omit both). Falling back to adhoc self-signed cert."
            )
            ssl_context = "adhoc"
        else:
            ssl_context = "adhoc"

    scheme = "https" if _dev_https else "http"
    print(
        f"[flask_app] Phones on LAN: open {scheme}://<YOUR-PC-IPv4>:{_port}/ "
        f"(IPv4 from `ipconfig`; not localhost). If the phone says 'timed out', "
        f"Windows Firewall may be blocking TCP {_port}: run scripts/allow_firewall_port.ps1 "
        "as Administrator once."
    )
    if _dev_https:
        if _ssl_cert and _ssl_key:
            print(
                "[flask_app] TLS: using FLASK_SSL_CERT / FLASK_SSL_KEY — "
                f"https://<this-machine-LAN-ip>:{_port}/demo/client-camera"
            )
        else:
            print(
                "[flask_app] TLS adhoc (self-signed) — browser will warn that the "
                "server cannot be verified. On your own LAN that's expected; tap "
                "Advanced / Details → proceed. Or set FLASK_SSL_CERT + FLASK_SSL_KEY "
                f"(see STREAMING.md) for trusted dev certs.\n"
                f" — https://<this-machine-LAN-ip>:{_port}/demo/client-camera"
            )
    app.run(
        host="0.0.0.0",
        port=_port,
        debug=True,
        ssl_context=ssl_context if _dev_https else None,
    )
