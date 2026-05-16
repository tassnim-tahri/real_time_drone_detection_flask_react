"""
Unified Real-Time Drone Detection & Analysis Backend
Combines live streaming detection with periodic VLM threat analysis
"""

from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import threading
import time
import json
import os
import uuid
import traceback
from collections import defaultdict, Counter
from werkzeug.utils import secure_filename
from datetime import datetime

# Import analysis functions from app.py
from app import (
    run_video_inference,
    analyze_crops_with_vlm,
    build_langchain_chain,
    generate_summary,
    generate_final_report,
    report_to_context_text,
    normalize_bbox,
    get_class_name,
    safe_float,
)

# =============================
# FLASK APP SETUP
# =============================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("uploads")
CROPS_FOLDER = Path("outputs/crops")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
CROPS_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Load YOLO model once
model = YOLO("model.pt")

# =============================
# GLOBAL STATE MANAGEMENT
# =============================
class StreamingDetectionState:
    """Manages real-time streaming detection state"""
    def __init__(self):
        self.frame = None
        self.fps = 0
        self.detection_count = 0
        self.objects_detected = defaultdict(int)
        self.confidence_threshold = 0.5
        self.lock = threading.Lock()
        self.last_time = time.time()
        self.frame_count = 0
        self.total_detections_lifetime = 0
        self.frames_with_detections = 0
        self.is_running = False
        self.camera_index = 0

class AnalysisState:
    """Manages periodic VLM analysis state"""
    def __init__(self):
        self.lock = threading.Lock()
        self.current_report = None
        self.report_text = None
        self.chain = None
        self.summary = None
        self.final_report = None
        self.is_analyzing = False
        self.last_analysis_time = None
        self.analysis_status = "idle"  # idle, analyzing, ready, error
        self.error_message = None
        self.vlm_analysis = None

stream_state = StreamingDetectionState()
analysis_state = AnalysisState()

# =============================
# REAL-TIME STREAMING FUNCTIONS
# =============================
def generate_frames():
    """Generate frames with real-time YOLO detection"""
    cap = cv2.VideoCapture(stream_state.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    while stream_state.is_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model(frame, verbose=False, conf=stream_state.confidence_threshold)
        annotated_frame = results[0].plot()
        
        # Extract detection info
        with stream_state.lock:
            stream_state.objects_detected.clear()
            stream_state.detection_count = 0
            frame_has_detection = False
            
            for box in results[0].boxes:
                conf = float(box.conf)
                if conf >= stream_state.confidence_threshold:
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]
                    stream_state.objects_detected[cls_name] += 1
                    stream_state.detection_count += 1
                    stream_state.total_detections_lifetime += 1
                    frame_has_detection = True
            
            if frame_has_detection:
                stream_state.frames_with_detections += 1
            
            # Calculate FPS
            stream_state.frame_count += 1
            current_time = time.time()
            if current_time - stream_state.last_time >= 1:
                stream_state.fps = stream_state.frame_count
                stream_state.frame_count = 0
                stream_state.last_time = current_time
            
            # Add stats overlay
            cv2.putText(annotated_frame, f'FPS: {stream_state.fps}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Detections: {stream_state.detection_count}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stream_state.frame = annotated_frame.copy()
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# =============================
# VIDEO FILE ANALYSIS FUNCTIONS
# =============================
def process_video_file_async(video_path: str, conf_threshold: float = 0.25, 
                             vlm_provider: str = "groq", 
                             vlm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
                             groq_api_key: str = None):
    """
    Asynchronously process a video file with full pipeline:
    1. YOLO detection + tracking
    2. Crop extraction
    3. VLM threat analysis
    4. LLM report generation
    """
    try:
        with analysis_state.lock:
            analysis_state.is_analyzing = True
            analysis_state.analysis_status = "analyzing"
            analysis_state.error_message = None
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        output_video_path = OUTPUT_FOLDER / f"annotated_{uuid.uuid4()}.mp4"
        crop_output_dir = CROPS_FOLDER / f"batch_{uuid.uuid4()}"
        crop_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[Analysis] Starting video inference on {video_path}")
        
        # Step 1: YOLO inference with tracking
        report = run_video_inference(
            model=model,
            video_path=video_path,
            output_video_path=output_video_path,
            conf_threshold=conf_threshold,
            device=None,
            crop_output_dir=crop_output_dir,
            vlm_min_yolo_conf=0.35,
            vlm_max_crops=12,
            tracking=True,
            tracker="bytetrack.yaml",
        )
        
        print(f"[Analysis] YOLO detection complete. Found {report['total_detections']} detections")
        
        # Step 2: VLM analysis on crops
        vlm_analysis = analyze_crops_with_vlm(
            candidates=report.get("vlm_candidate_crops", []),
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            groq_api_key=groq_api_key,
        )
        report["vlm_analysis"] = vlm_analysis
        
        print(f"[Analysis] VLM analysis complete. Analyzed {vlm_analysis.get('analyzed_crops', 0)} crops")
        
        # Step 3: Generate report text for LLM context
        report_text = report_to_context_text(report)
        
        # Step 4: Build LLM chain and generate summaries
        llm_model = "llama-3.3-70b-versatile" if vlm_provider == "groq" else "llama3.1"
        chain = build_langchain_chain("groq", llm_model, groq_api_key)
        
        summary, _summary_ok = generate_summary(chain, report_text, report)
        final_report, final_report_from_llm = generate_final_report(chain, report_text, report)
        
        print(f"[Analysis] Report generation complete")
        
        # Save outputs
        report_json_path = OUTPUT_FOLDER / f"report_{uuid.uuid4()}.json"
        report_json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        
        with analysis_state.lock:
            analysis_state.current_report = report
            analysis_state.report_text = report_text
            analysis_state.chain = chain
            analysis_state.summary = summary
            analysis_state.final_report = final_report
            analysis_state.vlm_analysis = vlm_analysis
            analysis_state.last_analysis_time = datetime.now().isoformat()
            analysis_state.analysis_status = "ready"
            analysis_state.is_analyzing = False
        
        print(f"[Analysis] Analysis pipeline complete!")
        
    except Exception as e:
        print(f"[Analysis Error] {str(e)}")
        traceback.print_exc()
        with analysis_state.lock:
            analysis_state.analysis_status = "error"
            analysis_state.error_message = str(e)
            analysis_state.is_analyzing = False

# =============================
# REST API ENDPOINTS
# =============================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Unified Real-Time Detection & Analysis",
        "streaming_active": stream_state.is_running,
        "analysis_status": analysis_state.analysis_status
    })

@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    """Start real-time camera streaming"""
    if stream_state.is_running:
        return jsonify({"error": "Stream already running"}), 400
    
    stream_state.is_running = True
    stream_state.total_detections_lifetime = 0
    stream_state.frames_with_detections = 0
    
    return jsonify({"success": True, "message": "Stream started"})

@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    """Stop real-time camera streaming"""
    stream_state.is_running = False
    return jsonify({"success": True, "message": "Stream stopped"})

@app.route("/video_feed")
def video_feed():
    """MJPEG video feed endpoint"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get real-time streaming stats"""
    with stream_state.lock:
        return jsonify({
            "fps": stream_state.fps,
            "detections": stream_state.detection_count,
            "total_detections": stream_state.total_detections_lifetime,
            "frames_with_detections": stream_state.frames_with_detections,
            "objects_detected": dict(stream_state.objects_detected),
            "confidence_threshold": stream_state.confidence_threshold,
            "streaming_active": stream_state.is_running
        })

@app.route("/api/set_confidence", methods=["POST"])
def set_confidence():
    """Update detection confidence threshold"""
    data = request.get_json() or {}
    threshold = float(data.get("threshold", 0.5))
    threshold = max(0.0, min(1.0, threshold))
    
    with stream_state.lock:
        stream_state.confidence_threshold = threshold
    
    return jsonify({
        "success": True,
        "threshold": stream_state.confidence_threshold
    })

@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Upload video file for analysis"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / f"{uuid.uuid4()}_{filename}"
    file.save(str(filepath))
    
    # Start async analysis
    conf_threshold = float(request.form.get("confidence", 0.25))
    vlm_provider = request.form.get("vlm_provider", "groq")
    vlm_model = request.form.get("vlm_model", "meta-llama/llama-4-scout-17b-16e-instruct")
    groq_api_key = request.form.get("groq_api_key") or os.getenv("GROQ_API_KEY")
    
    analysis_thread = threading.Thread(
        target=process_video_file_async,
        args=(str(filepath), conf_threshold, vlm_provider, vlm_model, groq_api_key),
        daemon=True
    )
    analysis_thread.start()
    
    return jsonify({
        "success": True,
        "message": "Video uploaded and analysis started",
        "video_path": str(filepath)
    })

@app.route("/api/analysis/status", methods=["GET"])
def analysis_status():
    """Get current analysis status"""
    with analysis_state.lock:
        status_data = {
            "status": analysis_state.analysis_status,
            "is_analyzing": analysis_state.is_analyzing,
            "last_analysis_time": analysis_state.last_analysis_time,
        }
        
        if analysis_state.error_message:
            status_data["error"] = analysis_state.error_message
        
        if analysis_state.analysis_status == "ready" and analysis_state.current_report:
            report = analysis_state.current_report
            status_data.update({
                "total_detections": report.get("total_detections", 0),
                "frames_with_detections": report.get("frames_with_detections", 0),
                "unique_tracks": report.get("tracking", {}).get("unique_tracks", 0),
                "duration_sec": report.get("duration_sec", 0),
                "fps": report.get("fps", 0),
            })
    
    return jsonify(status_data)

@app.route("/api/analysis/report", methods=["GET"])
def get_analysis_report():
    """Get full analysis report"""
    with analysis_state.lock:
        if analysis_state.analysis_status != "ready":
            return jsonify({
                "error": f"Analysis not ready. Status: {analysis_state.analysis_status}"
            }), 400
        
        if not analysis_state.final_report:
            return jsonify({"error": "No report available"}), 400
        
        return jsonify({
            "final_report": analysis_state.final_report,
            "summary": analysis_state.summary,
            "raw_report": analysis_state.current_report,
            "vlm_analysis": analysis_state.vlm_analysis
        })

@app.route("/api/analysis/summary", methods=["GET"])
def get_analysis_summary():
    """Get quick summary of analysis"""
    with analysis_state.lock:
        if not analysis_state.summary:
            return jsonify({"error": "No summary available"}), 400
        
        return jsonify({
            "summary": analysis_state.summary,
            "timestamp": analysis_state.last_analysis_time
        })

@app.route("/api/chat", methods=["POST"])
def chat():
    """Chat with LLM about analysis results"""
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        with analysis_state.lock:
            if analysis_state.analysis_status != "ready":
                return jsonify({
                    "error": "No analysis ready for chatting"
                }), 400
            
            if not analysis_state.chain:
                return jsonify({
                    "error": "LLM chain not available"
                }), 400
            
            chain = analysis_state.chain
            report_text = analysis_state.report_text
        
        try:
            response = chain.invoke({
                "report": report_text,
                "question": user_message
            })
            return jsonify({"response": str(response).strip()})
        except Exception as exc:
            return jsonify({
                "error": f"LLM error: {str(exc)}"
            }), 500
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/detection/quick", methods=["POST"])
def quick_detect():
    """Quick detection on uploaded image (no analysis)"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filepath = UPLOAD_FOLDER / secure_filename(file.filename)
    file.save(str(filepath))
    
    try:
        results = model(str(filepath), verbose=False)
        detections = []
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "label": cls_name,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2)
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

@app.route("/api/system/info", methods=["GET"])
def system_info():
    """Get system information"""
    return jsonify({
        "model_name": "YOLO (model.pt)",
        "model_loaded": True,
        "capabilities": [
            "Real-time streaming detection",
            "Video file analysis",
            "VLM threat analysis",
            "LLM report generation",
            "Interactive chat"
        ],
        "streaming_enabled": True,
        "vlm_enabled": True,
        "llm_enabled": True
    })

# =============================
# DEBUG & TESTING ENDPOINTS
# =============================

@app.route("/api/debug/reset_stats", methods=["POST"])
def reset_stats():
    """Reset all statistics (for testing)"""
    with stream_state.lock:
        stream_state.total_detections_lifetime = 0
        stream_state.frames_with_detections = 0
        stream_state.objects_detected.clear()
    
    with analysis_state.lock:
        analysis_state.current_report = None
        analysis_state.analysis_status = "idle"
    
    return jsonify({"success": True, "message": "Stats reset"})

# =============================
# MAIN
# =============================

if __name__ == "__main__":
    print("=" * 60)
    print("Unified Real-Time Detection & Analysis Backend")
    print("=" * 60)
    print(f"Starting server on 0.0.0.0:8001")
    print(f"Streaming endpoint: /video_feed")
    print(f"API base: /api/")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=8001, debug=False, threaded=True)
