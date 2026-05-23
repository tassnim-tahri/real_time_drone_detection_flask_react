from pathlib import Path
import os
import re
import traceback
import uuid

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

chain = None
reportText=None

def save_file(file):
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "AI + streaming",
            "weights": _WEIGHTS,
            "endpoints": {
                "detect": "/detect",
                "video_feed": "/video_feed",
                "stats": "/api/stats",
                "streaming_demo": "/demo/streaming",
            },
        }
    )


@app.route("/demo/streaming")
def streaming_demo_page():
    static_dir = Path(__file__).resolve().parent / "static"
    return send_from_directory(static_dir, "streaming_demo.html")

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
    app.run(host="0.0.0.0", port=8000, debug=True)
