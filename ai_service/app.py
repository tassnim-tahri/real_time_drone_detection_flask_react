from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("model.pt")


# ✅ Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI service running"})


# =========================
# 🎥 ORIGINAL (video)
# =========================
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)

    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "detected_object": name,
                "axe_x": x1,
                "axe_y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            })

    return jsonify({
        "success": True,
        "type": "video",
        "count": len(detections),
        "detections": detections
    })


# =========================
# 🖼️ NEW ENDPOINT (detect2)
# =========================
@app.route("/detect2", methods=["POST"])
def detect2():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    results = model(filepath)

    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": name,
                "confidence": round(conf, 3),
                "box": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })

    return jsonify({
        "success": True,
        "type": "image",
        "count": len(detections),
        "detections": detections,
        "filename": filename
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
