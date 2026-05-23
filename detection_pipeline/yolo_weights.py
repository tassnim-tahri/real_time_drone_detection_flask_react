"""
Resolve YOLO weights path for local runs when custom weights are not checked in.

Preference order:
1. YOLO_WEIGHTS_PATH environment variable (file path or Ultralytics hub name).
2. model.pt next to detection_pipeline cwd
3. weights/best.pt
4. Fallback to yolov8n.pt (downloaded automatically by Ultralytics on first use)
"""
from __future__ import annotations

import os
from pathlib import Path


def _pipeline_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_yolo_weights() -> str:
    env = os.environ.get("YOLO_WEIGHTS_PATH", "").strip()
    if env:
        return env

    root = _pipeline_root()
    candidates = [
        root / "model.pt",
        root / "weights" / "best.pt",
        root / "weights" / "model.pt",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    return "yolov8n.pt"
