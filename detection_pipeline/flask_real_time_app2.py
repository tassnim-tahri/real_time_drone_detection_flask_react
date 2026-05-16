from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import threading
import time
from collections import defaultdict

app = Flask(__name__)
CORS(app)
# YOLO Model Configuration
model = YOLO('model.pt')  # Use yolov8n, yolov8s, yolov8m, etc.

# Camera Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Global variables
class DetectionState:
    def __init__(self):
        self.frame = None
        self.fps = 0
        self.detection_count = 0
        self.objects_detected = defaultdict(int)
        self.confidence_threshold = 0.5
        self.lock = threading.Lock()
        self.last_time = time.time()
        self.frame_count = 0

state = DetectionState()

def generate_frames():
    """Generate frames with YOLO detection"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Extract detection info
        with state.lock:
            state.objects_detected.clear()
            state.detection_count = 0
            
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf >= state.confidence_threshold:
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]
                    state.objects_detected[cls_name] += 1
                    state.detection_count += 1
            
            # Calculate FPS
            state.frame_count += 1
            current_time = time.time()
            if current_time - state.last_time >= 1:
                state.fps = state.frame_count
                state.frame_count = 0
                state.last_time = current_time
            
            # Add FPS text to frame
            cv2.putText(annotated_frame, f'FPS: {state.fps}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Detections: {state.detection_count}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            state.frame = annotated_frame.copy()
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    with state.lock:
        return jsonify({
            'fps': state.fps,
            'detections': state.detection_count,
            'objects_detected': dict(state.objects_detected)
        })

@app.route('/api/set_confidence', methods=['POST'])
def set_confidence():
    data = request.get_json()
    threshold = data.get('threshold', 0.5)
    with state.lock:
        state.confidence_threshold = max(0, min(1, threshold))
    return jsonify({'success': True, 'threshold': state.confidence_threshold})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)