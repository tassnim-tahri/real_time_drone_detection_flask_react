from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import time
from real_time_app import (
    LIVE_STATS,
    LATEST_FRAME,
    launch_detection_thread
)
app = Flask(__name__)
CORS(app)
# START DETECTION AUTOMATICALLY

launch_detection_thread()

@app.route("/live_stats")
def live_stats():
    return jsonify(LIVE_STATS)

# VIDEO STREAM

def generate_frames():

    global LATEST_FRAME

    while True:

        if LATEST_FRAME is None:
            time.sleep(0.01)
            continue

        _, buffer = cv2.imencode(".jpg", LATEST_FRAME)

        frame_bytes = buffer.tobytes()
        
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )

@app.route("/video_feed")
def video_feed():
    from flask import Response
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# STOP DETECTION

@app.route("/stop")
def stop():
    LIVE_STATS["running"] = False
    return jsonify({
        "message": "Detection stopped"
    })

# MAIN

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8001,
        debug=False,
        use_reloader=False,
        threaded=True
    )