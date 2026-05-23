"""
Live camera MJPEG streaming with YOLO overlays.

Used by flask_app (port 8000) and can be reused by real_time_flask_app.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from typing import Any

import cv2
from flask import Flask, Response, jsonify, request

__all__ = [
    "StreamingState",
    "streaming_state",
    "register_streaming_routes",
    "clamp_float",
]


def clamp_float(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, float(value)))


class StreamingState:
    def __init__(self) -> None:
        self.frame = None
        self.fps = 0
        self.detection_count = 0
        self.objects_detected: defaultdict[str, int] = defaultdict(int)
        self.confidence_threshold = float(os.environ.get("STREAM_CONFIDENCE", "0.35"))
        self.lock = threading.Lock()
        self.last_time = time.time()
        self.frame_count = 0
        self.is_streaming = False
        self.total_detections_session = 0

    def camera_index(self) -> int:
        return int(os.environ.get("CAMERA_INDEX", "0"))


streaming_state = StreamingState()


def generate_frames(model: Any):
    """Multipart MJPEG generator; owns VideoCapture lifecycle per client connection."""
    cap = cv2.VideoCapture(streaming_state.camera_index())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps_target = int(os.environ.get("STREAM_FPS_TARGET", "30"))
    cap.set(cv2.CAP_PROP_FPS, fps_target)

    streaming_state.is_streaming = True

    try:
        while streaming_state.is_streaming:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False, conf=streaming_state.confidence_threshold)
            annotated_frame = results[0].plot()

            with streaming_state.lock:
                streaming_state.objects_detected.clear()
                streaming_state.detection_count = 0

                for box in results[0].boxes:
                    conf = float(box.conf)
                    if conf >= streaming_state.confidence_threshold:
                        cls_id = int(box.cls)
                        cls_name = model.names[cls_id]
                        streaming_state.objects_detected[cls_name] += 1
                        streaming_state.detection_count += 1
                        streaming_state.total_detections_session += 1

                streaming_state.frame_count += 1
                current_time = time.time()
                if current_time - streaming_state.last_time >= 1:
                    streaming_state.fps = streaming_state.frame_count
                    streaming_state.frame_count = 0
                    streaming_state.last_time = current_time

                cv2.putText(
                    annotated_frame,
                    f"FPS: {streaming_state.fps}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    f"Detections: {streaming_state.detection_count}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    f"Threshold: {streaming_state.confidence_threshold:.2f}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                streaming_state.frame = annotated_frame.copy()

            ok, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()
        streaming_state.is_streaming = False


def register_streaming_routes(app: Flask, model: Any) -> None:
    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_frames(model),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/stream/start", methods=["POST"])
    def start_stream():
        streaming_state.is_streaming = True
        return jsonify({"success": True, "message": "Streaming flag set (connect /video_feed to open camera)"})

    @app.route("/api/stream/stop", methods=["POST"])
    def stop_stream():
        streaming_state.is_streaming = False
        return jsonify({"success": True, "message": "Streaming stop requested"})

    @app.route("/api/stats")
    def get_streaming_stats():
        with streaming_state.lock:
            return jsonify(
                {
                    "fps": streaming_state.fps,
                    "detections": streaming_state.detection_count,
                    "objects_detected": dict(streaming_state.objects_detected),
                    "total_detections_session": streaming_state.total_detections_session,
                    "confidence_threshold": streaming_state.confidence_threshold,
                    "is_streaming": streaming_state.is_streaming,
                }
            )

    @app.route("/api/set_confidence", methods=["POST"])
    def set_confidence():
        try:
            data = request.get_json(force=True, silent=True) or {}
            threshold = float(data.get("threshold", 0.5))
            with streaming_state.lock:
                streaming_state.confidence_threshold = clamp_float(threshold, 0.0, 1.0)
            return jsonify({"success": True, "threshold": streaming_state.confidence_threshold})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route("/api/reset_session", methods=["POST"])
    def reset_session():
        with streaming_state.lock:
            streaming_state.total_detections_session = 0
            streaming_state.objects_detected.clear()
        return jsonify({"success": True, "message": "Session reset"})
