from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import threading
import time
from collections import defaultdict, Counter
from pathlib import Path
import json
import base64
import os
import re
from typing import Any, Optional

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# =============================
# INIT APP
# =============================
app = Flask(__name__)
CORS(app)

# YOLO Model Configuration
model = YOLO('model.pt')

# Camera Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# =============================
# GLOBAL STATE MANAGEMENT
# =============================
class StreamingState:
    """Manages real-time streaming stats"""
    def __init__(self):
        self.frame = None
        self.fps = 0
        self.detection_count = 0
        self.objects_detected = defaultdict(int)
        self.confidence_threshold = 0.5
        self.lock = threading.Lock()
        self.last_time = time.time()
        self.frame_count = 0
        self.is_streaming = False
        self.total_detections_session = 0
        self.tracking_enabled = True

class AnalysisState:
    """Manages video analysis pipeline results"""
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle, analyzing, completed, error
        self.progress = 0
        self.current_video = None
        self.total_frames = 0
        self.processed_frames = 0
        self.fps = 0
        self.total_detections = 0
        self.frames_with_detections = 0
        self.class_counts = {}
        self.timeline_data = {}
        self.top_events = []
        self.tracking_data = {
            "enabled": True,
            "unique_tracks": 0,
            "top_tracks": []
        }
        self.vlm_analysis = None
        self.report_text = ""
        self.summary = ""
        self.error_message = None

streaming_state = StreamingState()
analysis_state = AnalysisState()

# =============================
# REAL-TIME STREAMING
# =============================

def generate_frames():
    """Generate frames with YOLO detection for real-time streaming"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    streaming_state.is_streaming = True
    
    try:
        while streaming_state.is_streaming:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = model(frame, verbose=False, conf=streaming_state.confidence_threshold)
            annotated_frame = results[0].plot()
            
            # Extract detection info
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
                
                # Calculate FPS
                streaming_state.frame_count += 1
                current_time = time.time()
                if current_time - streaming_state.last_time >= 1:
                    streaming_state.fps = streaming_state.frame_count
                    streaming_state.frame_count = 0
                    streaming_state.last_time = current_time
                
                # Add text overlay
                cv2.putText(annotated_frame, f'FPS: {streaming_state.fps}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Detections: {streaming_state.detection_count}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Threshold: {streaming_state.confidence_threshold:.2f}', (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                streaming_state.frame = annotated_frame.copy()
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        streaming_state.is_streaming = False


# =============================
# VIDEO ANALYSIS PIPELINE
# =============================

def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback

def clamp_float(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, float(value)))

def get_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)

def normalize_bbox(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(xyxy[0]))))
    y1 = max(0, min(height - 1, int(round(xyxy[1]))))
    x2 = max(0, min(width, int(round(xyxy[2]))))
    y2 = max(0, min(height, int(round(xyxy[3]))))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)

def analyze_video_real_time(video_path: str, conf_threshold: float = 0.25):
    """Analyze video file with real-time progress updates"""
    try:
        with analysis_state.lock:
            analysis_state.status = "analyzing"
            analysis_state.progress = 0
            analysis_state.current_video = video_path
            analysis_state.error_message = None
        
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        fps = capture.get(cv2.CAP_PROP_FPS)
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
        
        with analysis_state.lock:
            analysis_state.total_frames = total_frames
            analysis_state.fps = fps
        
        print(f"Starting video analysis: {total_frames} frames at {fps} FPS")
        
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            
            # Run YOLO prediction
            prediction = model.predict(
                source=frame,
                conf=conf_threshold,
                device=None,
                verbose=False
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
                    label = get_class_name(model.names, class_id)
                    yolo_conf = round(float(confidence), 4)
                    
                    class_counts[label] += 1
                    detections_this_frame += 1
                    
                    second_mark = int(frame_index / fps)
                    timeline_counts[second_mark] += 1
                    
                    x1, y1, x2, y2 = normalize_bbox(xyxy, width, height)
                    
                    top_events.append({
                        "frame": frame_index,
                        "time_sec": round(frame_index / fps, 2),
                        "label": label,
                        "confidence": yolo_conf,
                        "bbox_xyxy": [x1, y1, x2, y2]
                    })
            
            frame_index += 1
            
            # Update progress
            with analysis_state.lock:
                analysis_state.processed_frames = frame_index
                analysis_state.progress = int((frame_index / total_frames) * 100) if total_frames > 0 else 0
                analysis_state.total_detections = sum(class_counts.values())
                analysis_state.frames_with_detections = len([1 for v in timeline_counts.values() if v > 0])
                analysis_state.class_counts = dict(class_counts)
                analysis_state.timeline_data = {str(k): v for k, v in sorted(timeline_counts.items())}
            
            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{total_frames} frames ({analysis_state.progress}%)")
        
        capture.release()
        
        # Sort top events by confidence
        top_events.sort(key=lambda x: x["confidence"], reverse=True)
        top_events = top_events[:50]
        
        with analysis_state.lock:
            analysis_state.top_events = top_events
            analysis_state.status = "completed"
            analysis_state.progress = 100
        
        print("Video analysis completed successfully")
        
    except Exception as e:
        print(f"Error during video analysis: {str(e)}")
        with analysis_state.lock:
            analysis_state.status = "error"
            analysis_state.error_message = str(e)


# =============================
# STREAMING ENDPOINTS
# =============================

@app.route('/video_feed')
def video_feed():
    """Stream live camera feed with YOLO detection"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start camera streaming"""
    streaming_state.is_streaming = True
    return jsonify({'success': True, 'message': 'Streaming started'})

@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop camera streaming"""
    streaming_state.is_streaming = False
    return jsonify({'success': True, 'message': 'Streaming stopped'})

@app.route('/api/stats')
def get_streaming_stats():
    """Get current streaming statistics"""
    with streaming_state.lock:
        return jsonify({
            'fps': streaming_state.fps,
            'detections': streaming_state.detection_count,
            'objects_detected': dict(streaming_state.objects_detected),
            'total_detections_session': streaming_state.total_detections_session,
            'confidence_threshold': streaming_state.confidence_threshold,
            'is_streaming': streaming_state.is_streaming
        })

@app.route('/api/set_confidence', methods=['POST'])
def set_confidence():
    """Update detection confidence threshold"""
    try:
        data = request.get_json()
        threshold = data.get('threshold', 0.5)
        
        with streaming_state.lock:
            streaming_state.confidence_threshold = clamp_float(threshold, 0.0, 1.0)
        
        return jsonify({
            'success': True,
            'threshold': streaming_state.confidence_threshold
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/reset_session', methods=['POST'])
def reset_session():
    """Reset session statistics"""
    with streaming_state.lock:
        streaming_state.total_detections_session = 0
        streaming_state.objects_detected.clear()
    
    return jsonify({'success': True, 'message': 'Session reset'})

# =============================
# ANALYSIS ENDPOINTS
# =============================

@app.route('/api/analysis/upload', methods=['POST'])
def upload_video_for_analysis():
    """Upload video for analysis and start processing"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        # Save file
        upload_dir = Path('analysis_uploads')
        upload_dir.mkdir(exist_ok=True)
        
        filename = f"{int(time.time())}_{file.filename}"
        filepath = upload_dir / filename
        file.save(str(filepath))
        
        # Get confidence threshold from request
        conf_threshold = request.form.get('confidence', 0.25, type=float)
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(
            target=analyze_video_real_time,
            args=(str(filepath), conf_threshold),
            daemon=True
        )
        analysis_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Analysis started',
            'video_file': filename
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analysis/status')
def get_analysis_status():
    """Get current analysis progress"""
    with analysis_state.lock:
        return jsonify({
            'status': analysis_state.status,
            'progress': analysis_state.progress,
            'current_video': analysis_state.current_video,
            'processed_frames': analysis_state.processed_frames,
            'total_frames': analysis_state.total_frames,
            'total_detections': analysis_state.total_detections,
            'frames_with_detections': analysis_state.frames_with_detections,
            'fps': analysis_state.fps,
            'error_message': analysis_state.error_message
        })

@app.route('/api/analysis/report')
def get_analysis_report():
    """Get full analysis report"""
    with analysis_state.lock:
        if analysis_state.status != "completed":
            return jsonify({'success': False, 'error': 'Analysis not completed'}), 400
        
        return jsonify({
            'success': True,
            'status': analysis_state.status,
            'fps': analysis_state.fps,
            'total_frames': analysis_state.total_frames,
            'total_detections': analysis_state.total_detections,
            'frames_with_detections': analysis_state.frames_with_detections,
            'detection_frame_ratio': round(
                analysis_state.frames_with_detections / analysis_state.total_frames, 4
            ) if analysis_state.total_frames > 0 else 0,
            'class_counts': analysis_state.class_counts,
            'timeline_detections_per_second': analysis_state.timeline_data,
            'top_confidence_events': analysis_state.top_events[:50],
            'tracking': analysis_state.tracking_data
        })

@app.route('/api/analysis/summary')
def get_analysis_summary():
    """Get quick summary of analysis"""
    with analysis_state.lock:
        if analysis_state.status != "completed":
            return jsonify({'success': False, 'error': 'Analysis not completed'}), 400
        
        # Generate summary text
        total_frames = analysis_state.total_frames
        duration_sec = total_frames / analysis_state.fps if analysis_state.fps > 0 else 0
        total_detections = analysis_state.total_detections
        frames_with_detections = analysis_state.frames_with_detections
        class_counts = analysis_state.class_counts
        
        if not class_counts:
            summary = f"Processed {total_frames} frames ({duration_sec:.1f}s). No detections found."
        else:
            top_class, top_count = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[0]
            ratio_pct = (frames_with_detections / total_frames * 100) if total_frames > 0 else 0
            summary = (
                f"Processed {total_frames} frames ({duration_sec:.1f}s), found {total_detections} detections in "
                f"{frames_with_detections} frames ({ratio_pct:.1f}%). Most frequent class: {top_class} ({top_count})."
            )
        
        return jsonify({
            'success': True,
            'summary': summary,
            'stats': {
                'total_frames': total_frames,
                'duration_sec': round(duration_sec, 2),
                'total_detections': total_detections,
                'frames_with_detections': frames_with_detections,
                'top_class': list(class_counts.keys())[0] if class_counts else None
            }
        })

@app.route('/api/analysis/reset', methods=['POST'])
def reset_analysis():
    """Reset analysis state"""
    with analysis_state.lock:
        analysis_state.status = "idle"
        analysis_state.progress = 0
        analysis_state.current_video = None
        analysis_state.total_frames = 0
        analysis_state.processed_frames = 0
        analysis_state.fps = 0
        analysis_state.total_detections = 0
        analysis_state.frames_with_detections = 0
        analysis_state.class_counts = {}
        analysis_state.timeline_data = {}
        analysis_state.top_events = []
        analysis_state.error_message = None
    
    return jsonify({'success': True, 'message': 'Analysis state reset'})

# =============================
# HEALTH CHECK
# =============================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Real-time Detection & Analysis',
        'model': 'YOLO',
        'streaming': streaming_state.is_streaming,
        'analysis': analysis_state.status
    })

# =============================
# ERROR HANDLERS
# =============================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# =============================
# MAIN
# =============================

if __name__ == '__main__':
    print("Starting Real-time Detection & Analysis Backend...")
    print("🎥 Streaming: http://192.168.1.17:8001/video_feed")
    print("📊 Stats: http://192.168.1.17:8001/api/stats")
    print("📈 Analysis: http://192.168.1.17:8001/api/analysis/status")
    
    app.run(
        host='0.0.0.0',
        port=8001,
        debug=False,
        threaded=True
    )
