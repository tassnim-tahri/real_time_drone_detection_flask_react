
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json
from enum import Enum


class DroneModel(str, Enum):
    """Available YOLO models"""
    NANO = "yolov8n.pt"
    SMALL = "yolov8s.pt"
    MEDIUM = "yolov8m.pt"
    LARGE = "yolov8l.pt"
    XLARGE = "yolov8x.pt"
    CUSTOM = "weights/model.pt"


class Tracker(str, Enum):
    """Available tracking algorithms"""
    BYTETRACK = "bytetrack.yaml"
    BOTSORT = "botsort.yaml"


class VLMProvider(str, Enum):
    """VLM providers"""
    GROQ = "groq"
    NONE = "none"


class LLMProvider(str, Enum):
    """LLM providers"""
    GROQ = "groq"
    OLLAMA = "ollama"
    NONE = "none"


@dataclass
class YOLOSettings:
    """YOLO Detection Configuration"""
    model: str = DroneModel.CUSTOM.value
    confidence_threshold: float = 0.25
    device: Optional[str] = None  # None for auto, '0' for GPU, 'cpu' for CPU
    img_size: int = 640
    
    def validate(self) -> bool:
        """Validate YOLO settings"""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if self.img_size not in [320, 416, 512, 640, 1024]:
            raise ValueError("img_size must be one of: 320, 416, 512, 640, 1024")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackingSettings:
    """Tracking Configuration"""
    enabled: bool = True
    tracker: str = Tracker.BYTETRACK.value
    max_age: int = 30  # Frames to keep track alive without detection
    min_hits: int = 3  # Detections needed to start track
    iou_threshold: float = 0.5  # IoU threshold for matching
    
    def validate(self) -> bool:
        """Validate tracking settings"""
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")
        if self.min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VLMSettings:
    """Vision Language Model (VLM) Configuration"""
    provider: str = VLMProvider.GROQ.value
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    max_crops: int = 12
    min_yolo_confidence: float = 0.35
    enable_payload_detection: bool = True
    
    def validate(self) -> bool:
        """Validate VLM settings"""
        if self.provider not in [p.value for p in VLMProvider]:
            raise ValueError(f"Invalid VLM provider: {self.provider}")
        if self.max_crops < 0:
            raise ValueError("max_crops must be >= 0")
        if not 0.0 <= self.min_yolo_confidence <= 1.0:
            raise ValueError("min_yolo_confidence must be between 0.0 and 1.0")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMSettings:
    """Large Language Model (LLM) Configuration"""
    provider: str = LLMProvider.GROQ.value
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    enable_chat: bool = True
    enable_reports: bool = True
    
    def validate(self) -> bool:
        """Validate LLM settings"""
        if self.provider not in [p.value for p in LLMProvider]:
            raise ValueError(f"Invalid LLM provider: {self.provider}")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StreamingSettings:
    """Real-time Streaming Configuration"""
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    fps: int = 30
    jpeg_quality: int = 85  # 1-100
    auto_adjust_quality: bool = True
    
    def validate(self) -> bool:
        """Validate streaming settings"""
        if self.camera_index < 0:
            raise ValueError("camera_index must be >= 0")
        if self.frame_width < 320 or self.frame_height < 240:
            raise ValueError("Minimum resolution is 320x240")
        if self.frame_width > 4096 or self.frame_height > 2160:
            raise ValueError("Maximum resolution is 4096x2160")
        if self.fps < 1 or self.fps > 60:
            raise ValueError("fps must be between 1 and 60")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisSettings:
    """Video Analysis Configuration"""
    max_frames_per_analysis: int = 0  # 0 = unlimited
    skip_frames: int = 0  # Process every Nth frame
    enable_visualization: bool = True
    save_crops: bool = True
    crop_quality: int = 90
    
    def validate(self) -> bool:
        """Validate analysis settings"""
        if self.max_frames_per_analysis < 0:
            raise ValueError("max_frames_per_analysis must be >= 0")
        if self.skip_frames < 0:
            raise ValueError("skip_frames must be >= 0")
        if not 1 <= self.crop_quality <= 100:
            raise ValueError("crop_quality must be between 1 and 100")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Settings:
    """Main Settings Manager - Central hub for all configuration"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize settings with optional config file"""
        self.config_file = config_file or Path("settings.json")
        
        # Initialize all setting groups
        self.yolo = YOLOSettings()
        self.tracking = TrackingSettings()
        self.vlm = VLMSettings()
        self.llm = LLMSettings()
        self.streaming = StreamingSettings()
        self.analysis = AnalysisSettings()
        
        # Load from file if exists
        if self.config_file.exists():
            self.load()
    
    # =====================================================================
    # YOLO SETTINGS FUNCTIONS
    # =====================================================================
    
    def set_yolo_confidence(self, threshold: float) -> bool:
        """
        Set YOLO confidence threshold
        
        Args:
            threshold: Confidence threshold (0.0-1.0)
        
        Returns:
            bool: Success
        """
        try:
            self.yolo.confidence_threshold = float(threshold)
            self.yolo.validate()
            self.save()
            return True
        except ValueError as e:
            raise ValueError(f"Invalid confidence threshold: {e}")
    
    def set_yolo_model(self, model: str) -> bool:
        """
        Set YOLO model
        
        Args:
            model: Model name from DroneModel enum or custom path
        
        Returns:
            bool: Success
        """
        try:
            # Check if it's a valid enum value
            if hasattr(DroneModel, model.upper().replace(".", "").replace("-", "").replace("/", "")):
                model = getattr(DroneModel, model.upper()).value
            
            self.yolo.model = model
            self.yolo.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid model: {e}")
    
    def set_yolo_device(self, device: Optional[str]) -> bool:
        """
        Set inference device
        
        Args:
            device: 'cpu', '0', '0,1' for GPU, or None for auto
        
        Returns:
            bool: Success
        """
        try:
            self.yolo.device = device
            self.yolo.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid device: {e}")
    
    def set_yolo_img_size(self, size: int) -> bool:
        """
        Set YOLO image size
        
        Args:
            size: 320, 416, 512, 640, or 1024
        
        Returns:
            bool: Success
        """
        try:
            self.yolo.img_size = int(size)
            self.yolo.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid image size: {e}")
    
    def get_yolo_settings(self) -> Dict[str, Any]:
        """Get all YOLO settings"""
        return self.yolo.to_dict()
    
    # =====================================================================
    # TRACKING SETTINGS FUNCTIONS
    # =====================================================================
    
    def set_tracking_enabled(self, enabled: bool) -> bool:
        """
        Enable/disable tracking
        
        Args:
            enabled: True to enable tracking
        
        Returns:
            bool: Success
        """
        try:
            self.tracking.enabled = bool(enabled)
            self.tracking.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error enabling tracking: {e}")
    
    def set_tracker_algorithm(self, tracker: str) -> bool:
        """
        Set tracking algorithm
        
        Args:
            tracker: 'bytetrack' or 'botsort'
        
        Returns:
            bool: Success
        """
        try:
            tracker_lower = tracker.lower()
            if tracker_lower == "bytetrack":
                self.tracking.tracker = Tracker.BYTETRACK.value
            elif tracker_lower == "botsort":
                self.tracking.tracker = Tracker.BOTSORT.value
            else:
                raise ValueError(f"Unknown tracker: {tracker}")
            
            self.tracking.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting tracker: {e}")
    
    def set_tracking_max_age(self, max_age: int) -> bool:
        """
        Set maximum frames to keep track alive without detection
        
        Args:
            max_age: Frames (minimum 1)
        
        Returns:
            bool: Success
        """
        try:
            self.tracking.max_age = int(max_age)
            self.tracking.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid max_age: {e}")
    
    def set_tracking_iou_threshold(self, threshold: float) -> bool:
        """
        Set IoU threshold for tracking
        
        Args:
            threshold: IoU threshold (0.0-1.0)
        
        Returns:
            bool: Success
        """
        try:
            self.tracking.iou_threshold = float(threshold)
            self.tracking.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid IoU threshold: {e}")
    
    def get_tracking_settings(self) -> Dict[str, Any]:
        """Get all tracking settings"""
        return self.tracking.to_dict()
    
    # =====================================================================
    # VLM SETTINGS FUNCTIONS
    # =====================================================================
    
    def set_vlm_provider(self, provider: str) -> bool:
        """
        Set VLM provider
        
        Args:
            provider: 'groq' or 'none'
        
        Returns:
            bool: Success
        """
        try:
            provider_lower = provider.lower()
            if provider_lower in [p.value for p in VLMProvider]:
                self.vlm.provider = provider_lower
            else:
                raise ValueError(f"Invalid provider: {provider}")
            
            self.vlm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting VLM provider: {e}")
    
    def set_vlm_model(self, model: str) -> bool:
        """
        Set VLM model
        
        Args:
            model: Model name (e.g., 'meta-llama/llama-4-scout-17b-16e-instruct')
        
        Returns:
            bool: Success
        """
        try:
            self.vlm.model = str(model)
            self.vlm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting VLM model: {e}")
    
    def set_vlm_max_crops(self, max_crops: int) -> bool:
        """
        Set maximum crops for VLM analysis
        
        Args:
            max_crops: Number of crops (0 for unlimited)
        
        Returns:
            bool: Success
        """
        try:
            self.vlm.max_crops = int(max_crops)
            self.vlm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid max_crops: {e}")
    
    def set_vlm_min_confidence(self, min_conf: float) -> bool:
        """
        Set minimum YOLO confidence for VLM crops
        
        Args:
            min_conf: Minimum confidence (0.0-1.0)
        
        Returns:
            bool: Success
        """
        try:
            self.vlm.min_yolo_confidence = float(min_conf)
            self.vlm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid min confidence: {e}")
    
    def set_vlm_payload_detection(self, enabled: bool) -> bool:
        """
        Enable/disable payload detection in VLM
        
        Args:
            enabled: True to enable
        
        Returns:
            bool: Success
        """
        try:
            self.vlm.enable_payload_detection = bool(enabled)
            self.vlm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting payload detection: {e}")
    
    def get_vlm_settings(self) -> Dict[str, Any]:
        """Get all VLM settings"""
        return self.vlm.to_dict()
    
    # =====================================================================
    # LLM SETTINGS FUNCTIONS
    # =====================================================================
    
    def set_llm_provider(self, provider: str) -> bool:
        """
        Set LLM provider
        
        Args:
            provider: 'groq', 'ollama', or 'none'
        
        Returns:
            bool: Success
        """
        try:
            provider_lower = provider.lower()
            if provider_lower in [p.value for p in LLMProvider]:
                self.llm.provider = provider_lower
            else:
                raise ValueError(f"Invalid provider: {provider}")
            
            self.llm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting LLM provider: {e}")
    
    def set_llm_model(self, model: str) -> bool:
        """
        Set LLM model
        
        Args:
            model: Model name
        
        Returns:
            bool: Success
        """
        try:
            self.llm.model = str(model)
            self.llm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting LLM model: {e}")
    
    def set_llm_temperature(self, temperature: float) -> bool:
        """
        Set LLM temperature
        
        Args:
            temperature: Temperature (0.0-1.0)
        
        Returns:
            bool: Success
        """
        try:
            self.llm.temperature = float(temperature)
            self.llm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid temperature: {e}")
    
    def set_llm_chat_enabled(self, enabled: bool) -> bool:
        """
        Enable/disable interactive chat
        
        Args:
            enabled: True to enable
        
        Returns:
            bool: Success
        """
        try:
            self.llm.enable_chat = bool(enabled)
            self.llm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting chat: {e}")
    
    def set_llm_reports_enabled(self, enabled: bool) -> bool:
        """
        Enable/disable automatic report generation
        
        Args:
            enabled: True to enable
        
        Returns:
            bool: Success
        """
        try:
            self.llm.enable_reports = bool(enabled)
            self.llm.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting reports: {e}")
    
    def get_llm_settings(self) -> Dict[str, Any]:
        """Get all LLM settings"""
        return self.llm.to_dict()
    
    # =====================================================================
    # STREAMING SETTINGS FUNCTIONS
    # =====================================================================
    
    def set_camera_index(self, index: int) -> bool:
        """
        Set camera index
        
        Args:
            index: Camera index (0 for default, 1 for USB, etc.)
        
        Returns:
            bool: Success
        """
        try:
            self.streaming.camera_index = int(index)
            self.streaming.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid camera index: {e}")
    
    def set_streaming_resolution(self, width: int, height: int) -> bool:
        """
        Set streaming resolution
        
        Args:
            width: Frame width (320-4096)
            height: Frame height (240-2160)
        
        Returns:
            bool: Success
        """
        try:
            self.streaming.frame_width = int(width)
            self.streaming.frame_height = int(height)
            self.streaming.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid resolution: {e}")
    
    def set_streaming_fps(self, fps: int) -> bool:
        """
        Set streaming FPS target
        
        Args:
            fps: Target FPS (1-60)
        
        Returns:
            bool: Success
        """
        try:
            self.streaming.fps = int(fps)
            self.streaming.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid FPS: {e}")
    
    def set_jpeg_quality(self, quality: int) -> bool:
        """
        Set JPEG compression quality
        
        Args:
            quality: Quality (1-100, lower = smaller file size)
        
        Returns:
            bool: Success
        """
        try:
            self.streaming.jpeg_quality = int(quality)
            self.streaming.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid JPEG quality: {e}")
    
    def get_streaming_settings(self) -> Dict[str, Any]:
        """Get all streaming settings"""
        return self.streaming.to_dict()
    
    # =====================================================================
    # ANALYSIS SETTINGS FUNCTIONS
    # =====================================================================
    
    def set_max_frames(self, max_frames: int) -> bool:
        """
        Set maximum frames to analyze
        
        Args:
            max_frames: Maximum frames (0 for unlimited)
        
        Returns:
            bool: Success
        """
        try:
            self.analysis.max_frames_per_analysis = int(max_frames)
            self.analysis.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid max_frames: {e}")
    
    def set_skip_frames(self, skip: int) -> bool:
        """
        Set frame skip (process every Nth frame)
        
        Args:
            skip: Skip count (0 = process all frames)
        
        Returns:
            bool: Success
        """
        try:
            self.analysis.skip_frames = int(skip)
            self.analysis.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Invalid skip_frames: {e}")
    
    def set_visualization_enabled(self, enabled: bool) -> bool:
        """
        Enable/disable visualization in output
        
        Args:
            enabled: True to enable
        
        Returns:
            bool: Success
        """
        try:
            self.analysis.enable_visualization = bool(enabled)
            self.analysis.validate()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error setting visualization: {e}")
    
    def get_analysis_settings(self) -> Dict[str, Any]:
        """Get all analysis settings"""
        return self.analysis.to_dict()
    
    # =====================================================================
    # GLOBAL SETTINGS FUNCTIONS
    # =====================================================================
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as dictionary"""
        return {
            "yolo": self.yolo.to_dict(),
            "tracking": self.tracking.to_dict(),
            "vlm": self.vlm.to_dict(),
            "llm": self.llm.to_dict(),
            "streaming": self.streaming.to_dict(),
            "analysis": self.analysis.to_dict(),
        }
    
    def save(self, filepath: Optional[Path] = None) -> bool:
        """
        Save settings to JSON file
        
        Args:
            filepath: Optional custom filepath
        
        Returns:
            bool: Success
        """
        try:
            path = filepath or self.config_file
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.get_all_settings(), f, indent=2)
            
            print(f"Settings saved to {path}")
            return True
        except Exception as e:
            raise IOError(f"Error saving settings: {e}")
    
    def load(self, filepath: Optional[Path] = None) -> bool:
        """
        Load settings from JSON file
        
        Args:
            filepath: Optional custom filepath
        
        Returns:
            bool: Success
        """
        try:
            path = filepath or self.config_file
            
            if not path.exists():
                print(f"Settings file not found: {path}")
                return False
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Load YOLO settings
            if 'yolo' in data:
                self.yolo = YOLOSettings(**data['yolo'])
            
            # Load tracking settings
            if 'tracking' in data:
                self.tracking = TrackingSettings(**data['tracking'])
            
            # Load VLM settings
            if 'vlm' in data:
                self.vlm = VLMSettings(**data['vlm'])
            
            # Load LLM settings
            if 'llm' in data:
                self.llm = LLMSettings(**data['llm'])
            
            # Load streaming settings
            if 'streaming' in data:
                self.streaming = StreamingSettings(**data['streaming'])
            
            # Load analysis settings
            if 'analysis' in data:
                self.analysis = AnalysisSettings(**data['analysis'])
            
            print(f"Settings loaded from {path}")
            return True
        except Exception as e:
            raise IOError(f"Error loading settings: {e}")
    
    def reset_to_defaults(self) -> bool:
        """Reset all settings to defaults"""
        try:
            self.yolo = YOLOSettings()
            self.tracking = TrackingSettings()
            self.vlm = VLMSettings()
            self.llm = LLMSettings()
            self.streaming = StreamingSettings()
            self.analysis = AnalysisSettings()
            self.save()
            return True
        except Exception as e:
            raise ValueError(f"Error resetting settings: {e}")
    
    def apply_preset(self, preset_name: str) -> bool:
        """
        Apply preset configuration
        
        Args:
            preset_name: 'fast', 'balanced', 'quality', 'privacy'
        
        Returns:
            bool: Success
        """
        try:
            presets = {
                "fast": {
                    "yolo": {"model": DroneModel.NANO.value, "confidence_threshold": 0.4, "img_size": 416},
                    "streaming": {"frame_width": 640, "frame_height": 480, "jpeg_quality": 75},
                    "analysis": {"skip_frames": 2},
                },
                "balanced": {
                    "yolo": {"model": DroneModel.SMALL.value, "confidence_threshold": 0.35, "img_size": 640},
                    "streaming": {"frame_width": 1280, "frame_height": 720, "jpeg_quality": 85},
                    "analysis": {"skip_frames": 0},
                },
                "quality": {
                    "yolo": {"model": DroneModel.MEDIUM.value, "confidence_threshold": 0.25, "img_size": 640},
                    "streaming": {"frame_width": 1920, "frame_height": 1080, "jpeg_quality": 95},
                    "analysis": {"skip_frames": 0},
                },
                "privacy": {
                    "vlm": {"max_crops": 0, "enable_payload_detection": False},
                    "llm": {"enable_chat": False},
                },
            }
            
            if preset_name not in presets:
                raise ValueError(f"Unknown preset: {preset_name}")
            
            preset = presets[preset_name]
            
            if 'yolo' in preset:
                for key, value in preset['yolo'].items():
                    setattr(self.yolo, key, value)
            
            if 'tracking' in preset:
                for key, value in preset['tracking'].items():
                    setattr(self.tracking, key, value)
            
            if 'vlm' in preset:
                for key, value in preset['vlm'].items():
                    setattr(self.vlm, key, value)
            
            if 'llm' in preset:
                for key, value in preset['llm'].items():
                    setattr(self.llm, key, value)
            
            if 'streaming' in preset:
                for key, value in preset['streaming'].items():
                    setattr(self.streaming, key, value)
            
            if 'analysis' in preset:
                for key, value in preset['analysis'].items():
                    setattr(self.analysis, key, value)
            
            self.save()
            print(f"Applied preset: {preset_name}")
            return True
        except Exception as e:
            raise ValueError(f"Error applying preset: {e}")


# Initialize global settings instance
global_settings = Settings()
