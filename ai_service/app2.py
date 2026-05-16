from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("model.pt")  # your trained YOLO model

@app.route("/detect", methods=["POST"])
def detect():
    if "video" not in request.files:
        return jsonify({"message": "No video uploaded"}), 400

    

    video = request.files["video"]
    filename = secure_filename(video.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    video.save(filepath)
    print("Saved file:", filepath)


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

    return jsonify(detections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    print("AI service is running on http://192.168.1.17:8000")
