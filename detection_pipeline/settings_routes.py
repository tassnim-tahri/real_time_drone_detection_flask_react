"""
Settings Management Routes for Flask App
Ready-to-integrate functions for app.py to handle all settings changes
"""

from flask import Blueprint, request, jsonify
from app_settings import global_settings
from typing import Dict, Any, Tuple

# Create Blueprint for settings routes
settings_bp = Blueprint('settings', __name__, url_prefix='/api/settings')


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def success_response(message: str, data: Any = None) -> Tuple[Dict, int]:
    """Generate success response"""
    response = {"success": True, "message": message}
    if data is not None:
        response["data"] = data
    return jsonify(response), 200


def error_response(message: str, error_detail: str = "") -> Tuple[Dict, int]:
    """Generate error response"""
    response = {
        "success": False,
        "error": message
    }
    if error_detail:
        response["detail"] = error_detail
    return jsonify(response), 400


# =====================================================================
# YOLO SETTINGS ENDPOINTS
# =====================================================================

@settings_bp.route('/yolo/confidence', methods=['POST'])
def set_yolo_confidence():
    """
    Set YOLO confidence threshold
    
    POST /api/settings/yolo/confidence
    {
        "threshold": 0.35
    }
    """
    try:
        data = request.get_json()
        threshold = data.get('threshold')
        
        if threshold is None:
            return error_response("Missing 'threshold' parameter")
        
        global_settings.set_yolo_confidence(threshold)
        return success_response(
            f"Confidence threshold set to {threshold}",
            {"confidence_threshold": threshold}
        )
    except ValueError as e:
        return error_response("Invalid confidence threshold", str(e))
    except Exception as e:
        return error_response("Error setting confidence", str(e))


@settings_bp.route('/yolo/model', methods=['POST'])
def set_yolo_model():
    """
    Set YOLO model
    
    POST /api/settings/yolo/model
    {
        "model": "yolov8s.pt"
    }
    
    Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, weights/model.pt
    """
    try:
        data = request.get_json()
        model = data.get('model')
        
        if not model:
            return error_response("Missing 'model' parameter")
        
        global_settings.set_yolo_model(model)
        return success_response(
            f"YOLO model set to {model}",
            {"model": model}
        )
    except ValueError as e:
        return error_response("Invalid model", str(e))
    except Exception as e:
        return error_response("Error setting model", str(e))


@settings_bp.route('/yolo/device', methods=['POST'])
def set_yolo_device():
    """
    Set inference device
    
    POST /api/settings/yolo/device
    {
        "device": "0"
    }
    
    Options: 'cpu', '0' (GPU 0), '0,1' (GPU 0,1), or null for auto
    """
    try:
        data = request.get_json()
        device = data.get('device')
        
        global_settings.set_yolo_device(device)
        return success_response(
            f"Device set to {device or 'auto'}",
            {"device": device}
        )
    except ValueError as e:
        return error_response("Invalid device", str(e))
    except Exception as e:
        return error_response("Error setting device", str(e))


@settings_bp.route('/yolo/img-size', methods=['POST'])
def set_yolo_img_size():
    """
    Set YOLO image size
    
    POST /api/settings/yolo/img-size
    {
        "size": 640
    }
    
    Options: 320, 416, 512, 640, 1024
    """
    try:
        data = request.get_json()
        size = data.get('size')
        
        if size is None:
            return error_response("Missing 'size' parameter")
        
        global_settings.set_yolo_img_size(size)
        return success_response(
            f"Image size set to {size}",
            {"img_size": size}
        )
    except ValueError as e:
        return error_response("Invalid image size", str(e))
    except Exception as e:
        return error_response("Error setting image size", str(e))


@settings_bp.route('/yolo', methods=['GET'])
def get_yolo_settings():
    """
    Get all YOLO settings
    
    GET /api/settings/yolo
    """
    try:
        settings = global_settings.get_yolo_settings()
        return success_response("YOLO settings retrieved", settings)
    except Exception as e:
        return error_response("Error retrieving YOLO settings", str(e))


# =====================================================================
# TRACKING SETTINGS ENDPOINTS
# =====================================================================

@settings_bp.route('/tracking/enabled', methods=['POST'])
def set_tracking_enabled():
    """
    Enable/disable tracking
    
    POST /api/settings/tracking/enabled
    {
        "enabled": true
    }
    """
    try:
        data = request.get_json()
        enabled = data.get('enabled')
        
        if enabled is None:
            return error_response("Missing 'enabled' parameter")
        
        global_settings.set_tracking_enabled(enabled)
        return success_response(
            f"Tracking {'enabled' if enabled else 'disabled'}",
            {"enabled": enabled}
        )
    except ValueError as e:
        return error_response("Error enabling tracking", str(e))
    except Exception as e:
        return error_response("Error setting tracking", str(e))


@settings_bp.route('/tracking/algorithm', methods=['POST'])
def set_tracker_algorithm():
    """
    Set tracking algorithm
    
    POST /api/settings/tracking/algorithm
    {
        "algorithm": "bytetrack"
    }
    
    Options: 'bytetrack', 'botsort'
    """
    try:
        data = request.get_json()
        algorithm = data.get('algorithm')
        
        if not algorithm:
            return error_response("Missing 'algorithm' parameter")
        
        global_settings.set_tracker_algorithm(algorithm)
        return success_response(
            f"Tracking algorithm set to {algorithm}",
            {"algorithm": algorithm}
        )
    except ValueError as e:
        return error_response("Invalid algorithm", str(e))
    except Exception as e:
        return error_response("Error setting algorithm", str(e))


@settings_bp.route('/tracking/max-age', methods=['POST'])
def set_tracking_max_age():
    """
    Set maximum frames to keep track alive without detection
    
    POST /api/settings/tracking/max-age
    {
        "max_age": 30
    }
    """
    try:
        data = request.get_json()
        max_age = data.get('max_age')
        
        if max_age is None:
            return error_response("Missing 'max_age' parameter")
        
        global_settings.set_tracking_max_age(max_age)
        return success_response(
            f"Max age set to {max_age}",
            {"max_age": max_age}
        )
    except ValueError as e:
        return error_response("Invalid max_age", str(e))
    except Exception as e:
        return error_response("Error setting max_age", str(e))


@settings_bp.route('/tracking/iou-threshold', methods=['POST'])
def set_tracking_iou():
    """
    Set IoU threshold for tracking
    
    POST /api/settings/tracking/iou-threshold
    {
        "threshold": 0.5
    }
    """
    try:
        data = request.get_json()
        threshold = data.get('threshold')
        
        if threshold is None:
            return error_response("Missing 'threshold' parameter")
        
        global_settings.set_tracking_iou_threshold(threshold)
        return success_response(
            f"IoU threshold set to {threshold}",
            {"iou_threshold": threshold}
        )
    except ValueError as e:
        return error_response("Invalid IoU threshold", str(e))
    except Exception as e:
        return error_response("Error setting IoU threshold", str(e))


@settings_bp.route('/tracking', methods=['GET'])
def get_tracking_settings():
    """
    Get all tracking settings
    
    GET /api/settings/tracking
    """
    try:
        settings = global_settings.get_tracking_settings()
        return success_response("Tracking settings retrieved", settings)
    except Exception as e:
        return error_response("Error retrieving tracking settings", str(e))


# =====================================================================
# VLM SETTINGS ENDPOINTS
# =====================================================================

@settings_bp.route('/vlm/provider', methods=['POST'])
def set_vlm_provider():
    """
    Set VLM provider
    
    POST /api/settings/vlm/provider
    {
        "provider": "groq"
    }
    
    Options: 'groq', 'none'
    """
    try:
        data = request.get_json()
        provider = data.get('provider')
        
        if not provider:
            return error_response("Missing 'provider' parameter")
        
        global_settings.set_vlm_provider(provider)
        return success_response(
            f"VLM provider set to {provider}",
            {"provider": provider}
        )
    except ValueError as e:
        return error_response("Invalid provider", str(e))
    except Exception as e:
        return error_response("Error setting provider", str(e))


@settings_bp.route('/vlm/model', methods=['POST'])
def set_vlm_model():
    """
    Set VLM model
    
    POST /api/settings/vlm/model
    {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct"
    }
    """
    try:
        data = request.get_json()
        model = data.get('model')
        
        if not model:
            return error_response("Missing 'model' parameter")
        
        global_settings.set_vlm_model(model)
        return success_response(
            f"VLM model set to {model}",
            {"model": model}
        )
    except ValueError as e:
        return error_response("Invalid model", str(e))
    except Exception as e:
        return error_response("Error setting model", str(e))


@settings_bp.route('/vlm/max-crops', methods=['POST'])
def set_vlm_max_crops():
    """
    Set maximum crops for VLM analysis
    
    POST /api/settings/vlm/max-crops
    {
        "max_crops": 12
    }
    """
    try:
        data = request.get_json()
        max_crops = data.get('max_crops')
        
        if max_crops is None:
            return error_response("Missing 'max_crops' parameter")
        
        global_settings.set_vlm_max_crops(max_crops)
        return success_response(
            f"Max crops set to {max_crops}",
            {"max_crops": max_crops}
        )
    except ValueError as e:
        return error_response("Invalid max_crops", str(e))
    except Exception as e:
        return error_response("Error setting max_crops", str(e))


@settings_bp.route('/vlm/min-confidence', methods=['POST'])
def set_vlm_min_confidence():
    """
    Set minimum YOLO confidence for VLM crops
    
    POST /api/settings/vlm/min-confidence
    {
        "min_confidence": 0.35
    }
    """
    try:
        data = request.get_json()
        min_conf = data.get('min_confidence')
        
        if min_conf is None:
            return error_response("Missing 'min_confidence' parameter")
        
        global_settings.set_vlm_min_confidence(min_conf)
        return success_response(
            f"Min confidence set to {min_conf}",
            {"min_confidence": min_conf}
        )
    except ValueError as e:
        return error_response("Invalid min confidence", str(e))
    except Exception as e:
        return error_response("Error setting min confidence", str(e))


@settings_bp.route('/vlm/payload-detection', methods=['POST'])
def set_vlm_payload_detection():
    """
    Enable/disable payload detection
    
    POST /api/settings/vlm/payload-detection
    {
        "enabled": true
    }
    """
    try:
        data = request.get_json()
        enabled = data.get('enabled')
        
        if enabled is None:
            return error_response("Missing 'enabled' parameter")
        
        global_settings.set_vlm_payload_detection(enabled)
        return success_response(
            f"Payload detection {'enabled' if enabled else 'disabled'}",
            {"enabled": enabled}
        )
    except ValueError as e:
        return error_response("Error setting payload detection", str(e))
    except Exception as e:
        return error_response("Error with payload detection", str(e))


@settings_bp.route('/vlm', methods=['GET'])
def get_vlm_settings():
    """
    Get all VLM settings
    
    GET /api/settings/vlm
    """
    try:
        settings = global_settings.get_vlm_settings()
        return success_response("VLM settings retrieved", settings)
    except Exception as e:
        return error_response("Error retrieving VLM settings", str(e))


# =====================================================================
# LLM SETTINGS ENDPOINTS
# =====================================================================

@settings_bp.route('/llm/provider', methods=['POST'])
def set_llm_provider():
    """
    Set LLM provider
    
    POST /api/settings/llm/provider
    {
        "provider": "groq"
    }
    
    Options: 'groq', 'ollama', 'none'
    """
    try:
        data = request.get_json()
        provider = data.get('provider')
        
        if not provider:
            return error_response("Missing 'provider' parameter")
        
        global_settings.set_llm_provider(provider)
        return success_response(
            f"LLM provider set to {provider}",
            {"provider": provider}
        )
    except ValueError as e:
        return error_response("Invalid provider", str(e))
    except Exception as e:
        return error_response("Error setting provider", str(e))


@settings_bp.route('/llm/model', methods=['POST'])
def set_llm_model():
    """
    Set LLM model
    
    POST /api/settings/llm/model
    {
        "model": "llama-3.3-70b-versatile"
    }
    """
    try:
        data = request.get_json()
        model = data.get('model')
        
        if not model:
            return error_response("Missing 'model' parameter")
        
        global_settings.set_llm_model(model)
        return success_response(
            f"LLM model set to {model}",
            {"model": model}
        )
    except ValueError as e:
        return error_response("Invalid model", str(e))
    except Exception as e:
        return error_response("Error setting model", str(e))


@settings_bp.route('/llm/temperature', methods=['POST'])
def set_llm_temperature():
    """
    Set LLM temperature
    
    POST /api/settings/llm/temperature
    {
        "temperature": 0.0
    }
    """
    try:
        data = request.get_json()
        temperature = data.get('temperature')
        
        if temperature is None:
            return error_response("Missing 'temperature' parameter")
        
        global_settings.set_llm_temperature(temperature)
        return success_response(
            f"Temperature set to {temperature}",
            {"temperature": temperature}
        )
    except ValueError as e:
        return error_response("Invalid temperature", str(e))
    except Exception as e:
        return error_response("Error setting temperature", str(e))


@settings_bp.route('/llm/chat', methods=['POST'])
def set_llm_chat():
    """
    Enable/disable interactive chat
    
    POST /api/settings/llm/chat
    {
        "enabled": true
    }
    """
    try:
        data = request.get_json()
        enabled = data.get('enabled')
        
        if enabled is None:
            return error_response("Missing 'enabled' parameter")
        
        global_settings.set_llm_chat_enabled(enabled)
        return success_response(
            f"Chat {'enabled' if enabled else 'disabled'}",
            {"enabled": enabled}
        )
    except ValueError as e:
        return error_response("Error setting chat", str(e))
    except Exception as e:
        return error_response("Error with chat", str(e))


@settings_bp.route('/llm/reports', methods=['POST'])
def set_llm_reports():
    """
    Enable/disable automatic report generation
    
    POST /api/settings/llm/reports
    {
        "enabled": true
    }
    """
    try:
        data = request.get_json()
        enabled = data.get('enabled')
        
        if enabled is None:
            return error_response("Missing 'enabled' parameter")
        
        global_settings.set_llm_reports_enabled(enabled)
        return success_response(
            f"Reports {'enabled' if enabled else 'disabled'}",
            {"enabled": enabled}
        )
    except ValueError as e:
        return error_response("Error setting reports", str(e))
    except Exception as e:
        return error_response("Error with reports", str(e))


@settings_bp.route('/llm', methods=['GET'])
def get_llm_settings():
    """
    Get all LLM settings
    
    GET /api/settings/llm
    """
    try:
        settings = global_settings.get_llm_settings()
        return success_response("LLM settings retrieved", settings)
    except Exception as e:
        return error_response("Error retrieving LLM settings", str(e))


# =====================================================================
# STREAMING SETTINGS ENDPOINTS
# =====================================================================

@settings_bp.route('/streaming/camera', methods=['POST'])
def set_camera_index():
    """
    Set camera index
    
    POST /api/settings/streaming/camera
    {
        "index": 0
    }
    """
    try:
        data = request.get_json()
        index = data.get('index')
        
        if index is None:
            return error_response("Missing 'index' parameter")
        
        global_settings.set_camera_index(index)
        return success_response(
            f"Camera set to index {index}",
            {"camera_index": index}
        )
    except ValueError as e:
        return error_response("Invalid camera index", str(e))
    except Exception as e:
        return error_response("Error setting camera", str(e))


@settings_bp.route('/streaming/resolution', methods=['POST'])
def set_streaming_resolution():
    """
    Set streaming resolution
    
    POST /api/settings/streaming/resolution
    {
        "width": 1280,
        "height": 720
    }
    """
    try:
        data = request.get_json()
        width = data.get('width')
        height = data.get('height')
        
        if width is None or height is None:
            return error_response("Missing 'width' or 'height' parameters")
        
        global_settings.set_streaming_resolution(width, height)
        return success_response(
            f"Resolution set to {width}x{height}",
            {"width": width, "height": height}
        )
    except ValueError as e:
        return error_response("Invalid resolution", str(e))
    except Exception as e:
        return error_response("Error setting resolution", str(e))


@settings_bp.route('/streaming/fps', methods=['POST'])
def set_streaming_fps():
    """
    Set streaming FPS target
    
    POST /api/settings/streaming/fps
    {
        "fps": 30
    }
    """
    try:
        data = request.get_json()
        fps = data.get('fps')
        
        if fps is None:
            return error_response("Missing 'fps' parameter")
        
        global_settings.set_streaming_fps(fps)
        return success_response(
            f"FPS set to {fps}",
            {"fps": fps}
        )
    except ValueError as e:
        return error_response("Invalid FPS", str(e))
    except Exception as e:
        return error_response("Error setting FPS", str(e))


@settings_bp.route('/streaming/jpeg-quality', methods=['POST'])
def set_jpeg_quality():
    """
    Set JPEG compression quality
    
    POST /api/settings/streaming/jpeg-quality
    {
        "quality": 85
    }
    """
    try:
        data = request.get_json()
        quality = data.get('quality')
        
        if quality is None:
            return error_response("Missing 'quality' parameter")
        
        global_settings.set_jpeg_quality(quality)
        return success_response(
            f"JPEG quality set to {quality}",
            {"jpeg_quality": quality}
        )
    except ValueError as e:
        return error_response("Invalid JPEG quality", str(e))
    except Exception as e:
        return error_response("Error setting JPEG quality", str(e))


@settings_bp.route('/streaming', methods=['GET'])
def get_streaming_settings():
    """
    Get all streaming settings
    
    GET /api/settings/streaming
    """
    try:
        settings = global_settings.get_streaming_settings()
        return success_response("Streaming settings retrieved", settings)
    except Exception as e:
        return error_response("Error retrieving streaming settings", str(e))


# =====================================================================
# ANALYSIS SETTINGS ENDPOINTS
# =====================================================================

@settings_bp.route('/analysis/max-frames', methods=['POST'])
def set_max_frames():
    """
    Set maximum frames to analyze
    
    POST /api/settings/analysis/max-frames
    {
        "max_frames": 0
    }
    
    (0 = unlimited)
    """
    try:
        data = request.get_json()
        max_frames = data.get('max_frames')
        
        if max_frames is None:
            return error_response("Missing 'max_frames' parameter")
        
        global_settings.set_max_frames(max_frames)
        return success_response(
            f"Max frames set to {max_frames or 'unlimited'}",
            {"max_frames": max_frames}
        )
    except ValueError as e:
        return error_response("Invalid max_frames", str(e))
    except Exception as e:
        return error_response("Error setting max_frames", str(e))


@settings_bp.route('/analysis/skip-frames', methods=['POST'])
def set_skip_frames():
    """
    Set frame skip (process every Nth frame)
    
    POST /api/settings/analysis/skip-frames
    {
        "skip": 0
    }
    
    (0 = process all frames)
    """
    try:
        data = request.get_json()
        skip = data.get('skip')
        
        if skip is None:
            return error_response("Missing 'skip' parameter")
        
        global_settings.set_skip_frames(skip)
        return success_response(
            f"Skip frames set to {skip}",
            {"skip_frames": skip}
        )
    except ValueError as e:
        return error_response("Invalid skip_frames", str(e))
    except Exception as e:
        return error_response("Error setting skip_frames", str(e))


@settings_bp.route('/analysis/visualization', methods=['POST'])
def set_visualization():
    """
    Enable/disable visualization in output
    
    POST /api/settings/analysis/visualization
    {
        "enabled": true
    }
    """
    try:
        data = request.get_json()
        enabled = data.get('enabled')
        
        if enabled is None:
            return error_response("Missing 'enabled' parameter")
        
        global_settings.set_visualization_enabled(enabled)
        return success_response(
            f"Visualization {'enabled' if enabled else 'disabled'}",
            {"enabled": enabled}
        )
    except ValueError as e:
        return error_response("Error setting visualization", str(e))
    except Exception as e:
        return error_response("Error with visualization", str(e))


@settings_bp.route('/analysis', methods=['GET'])
def get_analysis_settings():
    """
    Get all analysis settings
    
    GET /api/settings/analysis
    """
    try:
        settings = global_settings.get_analysis_settings()
        return success_response("Analysis settings retrieved", settings)
    except Exception as e:
        return error_response("Error retrieving analysis settings", str(e))


# =====================================================================
# GLOBAL SETTINGS ENDPOINTS
# =====================================================================

@settings_bp.route('/all', methods=['GET'])
def get_all_settings():
    """
    Get all settings
    
    GET /api/settings/all
    """
    try:
        settings = global_settings.get_all_settings()
        return success_response("All settings retrieved", settings)
    except Exception as e:
        return error_response("Error retrieving settings", str(e))


@settings_bp.route('/reset', methods=['POST'])
def reset_settings():
    """
    Reset all settings to defaults
    
    POST /api/settings/reset
    """
    try:
        global_settings.reset_to_defaults()
        return success_response(
            "All settings reset to defaults",
            global_settings.get_all_settings()
        )
    except ValueError as e:
        return error_response("Error resetting settings", str(e))
    except Exception as e:
        return error_response("Error during reset", str(e))


@settings_bp.route('/preset/<preset_name>', methods=['POST'])
def apply_preset(preset_name: str):
    """
    Apply preset configuration
    
    POST /api/settings/preset/{preset_name}
    
    Options: 'fast', 'balanced', 'quality', 'privacy'
    """
    try:
        global_settings.apply_preset(preset_name)
        return success_response(
            f"Applied preset: {preset_name}",
            global_settings.get_all_settings()
        )
    except ValueError as e:
        return error_response("Invalid preset", str(e))
    except Exception as e:
        return error_response("Error applying preset", str(e))


@settings_bp.route('/save', methods=['POST'])
def save_settings():
    """
    Save current settings to file
    
    POST /api/settings/save
    """
    try:
        global_settings.save()
        return success_response(
            "Settings saved successfully"
        )
    except IOError as e:
        return error_response("Error saving settings", str(e))
    except Exception as e:
        return error_response("Error during save", str(e))


@settings_bp.route('/load', methods=['POST'])
def load_settings():
    """
    Load settings from file
    
    POST /api/settings/load
    """
    try:
        global_settings.load()
        return success_response(
            "Settings loaded successfully",
            global_settings.get_all_settings()
        )
    except IOError as e:
        return error_response("Error loading settings", str(e))
    except Exception as e:
        return error_response("Error during load", str(e))


# =====================================================================
# HOW TO INTEGRATE INTO app.py
# =====================================================================

"""
In your Flask app (app.py or flask_app.py), add these lines:

from settings_routes import settings_bp

app = Flask(__name__)
app.register_blueprint(settings_bp)

That's it! All settings endpoints will be available at /api/settings/*
"""
