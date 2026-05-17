from __future__ import annotations

import argparse
import base64
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import cv2
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import torch
from ultralytics import YOLO

"""try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None"""
from langchain_community.chat_models import ChatOllama

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from groq import Groq
except ImportError:
    Groq = None



DRONE_ANALYSIS_PROMPT = """
You are a drone threat assessment analyst. Analyze this drone image carefully.

Return your analysis ONLY as a JSON object with this exact structure:
{
  "drone_type": "military | commercial | DIY | unknown",
  "payload_detected": true/false,
  "payload_description": "description or null",
  "payload_type": "camera | weapon | package | sensor | unknown | none",
  "estimated_size": "small | medium | large",
  "threat_level": "none | low | medium | high | critical",
  "threat_reasoning": "brief explanation",
  "notable_features": ["list", "of", "features"],
  "confidence": 0.0-1.0
}
""".strip()


THREAT_LEVEL_VALUES: dict[str, float] = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}


def resolve_path(raw_path: str, root_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def resolve_default_pt_weights(root_dir: Path) -> Path:
    best_pt = (root_dir / "weights" / "model.pt").resolve()
    if best_pt.exists():
        return best_pt

    candidates = sorted((root_dir / "weights").glob("*.pt"))
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(
        "No .pt weights found in weights/. Provide --weights explicitly or add a .pt file to weights/."
    )


def resolve_model_source(weights_arg: str, root_dir: Path) -> str:
    candidate_path = resolve_path(weights_arg, root_dir)
    if candidate_path.exists():
        return str(candidate_path)

    # Allow Ultralytics model aliases (for example yolo26n.pt) to be downloaded/loaded by name.
    if re.match(r"^yolo\d+[a-z0-9_-]*\.pt$", weights_arg, flags=re.IGNORECASE):
        return weights_arg

    raise FileNotFoundError(
        f"Weights not found: {candidate_path}. Provide a valid local path or a YOLO model alias like yolo26n.pt."
    )


def get_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def clamp_float(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, float(value)))


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalize_bbox(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(xyxy[0]))))
    y1 = max(0, min(height - 1, int(round(xyxy[1]))))
    x2 = max(0, min(width, int(round(xyxy[2]))))
    y2 = max(0, min(height, int(round(xyxy[3]))))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def cleanup_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def parse_json_payload(text: str) -> dict[str, Any]:
    cleaned = cleanup_json_text(text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}

    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return {}

    return {}


def normalize_drone_type(raw_value: Any) -> str:
    value = str(raw_value or "unknown").strip().lower()
    if value == "diy":
        return "DIY"
    if value in {"military", "commercial", "unknown"}:
        return value
    return "unknown"


def normalize_vlm_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    payload_detected = bool(raw.get("payload_detected", False))
    payload_description = raw.get("payload_description")
    payload_type = str(raw.get("payload_type", "unknown")).strip().lower()
    estimated_size = str(raw.get("estimated_size", "medium")).strip().lower()
    threat_level = str(raw.get("threat_level", "none")).strip().lower()
    threat_reasoning = str(raw.get("threat_reasoning", "No reasoning provided.")).strip()
    confidence = clamp_float(safe_float(raw.get("confidence", 0.0), 0.0), 0.0, 1.0)

    if payload_type not in {"camera", "weapon", "package", "sensor", "unknown", "none"}:
        payload_type = "unknown"

    if estimated_size not in {"small", "medium", "large"}:
        estimated_size = "medium"

    if threat_level not in THREAT_LEVEL_VALUES:
        threat_level = "none"

    notable_features_raw = raw.get("notable_features", [])
    notable_features: list[str] = []
    if isinstance(notable_features_raw, list):
        notable_features = [str(item).strip() for item in notable_features_raw if str(item).strip()]

    if not payload_detected:
        payload_description = None
        if payload_type != "none":
            payload_type = "none"

    if payload_description is not None:
        payload_description = str(payload_description).strip() or None

    return {
        "drone_type": normalize_drone_type(raw.get("drone_type", "unknown")),
        "payload_detected": payload_detected,
        "payload_description": payload_description,
        "payload_type": payload_type,
        "estimated_size": estimated_size,
        "threat_level": threat_level,
        "threat_reasoning": threat_reasoning,
        "notable_features": notable_features,
        "confidence": round(confidence, 4),
    }


def score_to_risk_band(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.45:
        return "medium"
    if score >= 0.25:
        return "low"
    return "none"


def compute_unified_score(yolo_confidence: float, threat_level: str, vlm_confidence: float) -> float:
    threat_value = THREAT_LEVEL_VALUES.get(str(threat_level).lower(), 0.0)
    threat_component = threat_value * clamp_float(vlm_confidence, 0.0, 1.0)
    combined = 0.55 * clamp_float(yolo_confidence, 0.0, 1.0) + 0.45 * threat_component
    return round(clamp_float(combined, 0.0, 1.0), 4)


def run_video_inference(
    model: YOLO,
    video_path: Path,
    output_video_path: Path,
    conf_threshold: float,
    device: Optional[str],
    crop_output_dir: Path,
    vlm_min_yolo_conf: float,
    vlm_max_crops: int,
    tracking: bool,
    tracker: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video size. Could not read width/height from the input video.")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_video_path}")

    class_counts: Counter[str] = Counter()
    timeline_counts: Counter[int] = Counter()
    per_frame_counts: list[int] = []
    top_events: list[dict[str, Any]] = []

    # Track the strongest detection in each second to avoid near-duplicate crops.
    candidate_by_second: dict[int, dict[str, Any]] = {}
    # If tracking is enabled, keep one strongest crop per track ID.
    candidate_by_track: dict[int, dict[str, Any]] = {}

    track_records: dict[int, dict[str, Any]] = {}
    tracked_detection_count = 0

    frame_index = 0
    print("Running inference on video...")
    if tracking:
        print(f"Tracking is enabled using tracker config: {tracker}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if tracking:
            prediction = model.track(
                source=frame,
                conf=conf_threshold,
                device=device,
                verbose=False,
                persist=True,
                tracker=tracker,
            )
        else:
            prediction = model.predict(source=frame, conf=conf_threshold, device=device, verbose=False)

        result = prediction[0]

        detections_this_frame = 0
        boxes = result.boxes

        if boxes is not None and boxes.cls is not None and boxes.conf is not None and boxes.xyxy is not None:
            class_ids = boxes.cls.tolist()
            confidences = boxes.conf.tolist()
            xyxy_values = boxes.xyxy.tolist()

            if boxes.id is not None:
                track_ids: list[Optional[int]] = [
                    int(track_id_value) if track_id_value is not None else None
                    for track_id_value in boxes.id.tolist()
                ]
            else:
                track_ids = [None] * len(class_ids)

            for class_id_float, confidence, xyxy, track_id in zip(class_ids, confidences, xyxy_values, track_ids):
                class_id = int(class_id_float)
                label = get_class_name(model.names, class_id)
                yolo_conf = round(float(confidence), 4)

                class_counts[label] += 1
                detections_this_frame += 1

                second_mark = int(frame_index / fps)
                timeline_counts[second_mark] += 1

                x1, y1, x2, y2 = normalize_bbox(xyxy, width, height)

                top_events.append(
                    {
                        "frame": frame_index,
                        "time_sec": round(frame_index / fps, 2),
                        "label": label,
                        "track_id": track_id,
                        "confidence": yolo_conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )

                if track_id is not None:
                    tracked_detection_count += 1
                    existing_record = track_records.get(track_id)
                    if existing_record is None:
                        track_records[track_id] = {
                            "track_id": track_id,
                            "label": label,
                            "first_frame": frame_index,
                            "last_frame": frame_index,
                            "detection_count": 1,
                            "max_confidence": yolo_conf,
                        }
                    else:
                        existing_record["last_frame"] = frame_index
                        existing_record["detection_count"] = int(existing_record["detection_count"]) + 1
                        existing_record["max_confidence"] = max(
                            safe_float(existing_record.get("max_confidence"), 0.0),
                            yolo_conf,
                        )

                if vlm_max_crops <= 0 or yolo_conf < vlm_min_yolo_conf:
                    continue

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                candidate_payload = {
                    "frame": frame_index,
                    "time_sec": round(frame_index / fps, 2),
                    "label": label,
                    "track_id": track_id,
                    "yolo_confidence": yolo_conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "_crop_image": crop.copy(),
                }

                if tracking and track_id is not None:
                    existing_track_candidate = candidate_by_track.get(track_id)
                    if (
                        existing_track_candidate is None
                        or yolo_conf > safe_float(existing_track_candidate.get("yolo_confidence"), 0.0)
                    ):
                        candidate_by_track[track_id] = candidate_payload
                else:
                    existing_second_candidate = candidate_by_second.get(second_mark)
                    if (
                        existing_second_candidate is None
                        or yolo_conf > safe_float(existing_second_candidate.get("yolo_confidence"), 0.0)
                    ):
                        candidate_by_second[second_mark] = candidate_payload

        per_frame_counts.append(detections_this_frame)
        annotated_frame = result.plot()
        writer.write(annotated_frame)

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"Processed {frame_index} frames...")

    capture.release()
    writer.release()

    top_events.sort(key=lambda item: item["confidence"], reverse=True)
    top_events = top_events[:50]

    if candidate_by_track:
        candidate_pool = list(candidate_by_track.values())
    else:
        candidate_pool = list(candidate_by_second.values())

    selected_candidates = sorted(
        candidate_pool,
        key=lambda item: safe_float(item.get("yolo_confidence"), 0.0),
        reverse=True,
    )[: max(vlm_max_crops, 0)]

    saved_candidates: list[dict[str, Any]] = []
    if selected_candidates:
        crop_output_dir.mkdir(parents=True, exist_ok=True)

    for idx, candidate in enumerate(selected_candidates, start=1):
        crop_image = candidate.pop("_crop_image", None)
        if crop_image is None:
            continue

        safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", str(candidate.get("label", "drone")))
        confidence_tag = str(candidate.get("yolo_confidence", 0.0)).replace(".", "p")
        track_tag = f"_t{candidate.get('track_id')}" if candidate.get("track_id") is not None else ""
        crop_file = crop_output_dir / (
            f"crop_{idx:03d}_{safe_label}{track_tag}_f{candidate.get('frame', 0)}_c{confidence_tag}.jpg"
        )

        write_ok = cv2.imwrite(
            str(crop_file),
            crop_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not write_ok:
            continue

        candidate["crop_path"] = str(crop_file)
        saved_candidates.append(candidate)

    total_detections = int(sum(per_frame_counts))
    frames_with_detections = int(sum(1 for count in per_frame_counts if count > 0))

    track_summaries: list[dict[str, Any]] = []
    for record in track_records.values():
        first_frame = int(record.get("first_frame", 0))
        last_frame = int(record.get("last_frame", first_frame))
        duration_frames = max(1, last_frame - first_frame + 1)
        duration_sec = round(duration_frames / fps, 2)

        track_summaries.append(
            {
                "track_id": int(record.get("track_id", -1)),
                "label": str(record.get("label", "drone")),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "duration_frames": duration_frames,
                "duration_sec": duration_sec,
                "detection_count": int(record.get("detection_count", 0)),
                "max_confidence": round(safe_float(record.get("max_confidence", 0.0)), 4),
            }
        )

    track_summaries.sort(
        key=lambda item: (
            int(item.get("duration_frames", 0)),
            safe_float(item.get("max_confidence", 0.0)),
        ),
        reverse=True,
    )

    avg_track_duration_frames = (
        round(sum(int(item.get("duration_frames", 0)) for item in track_summaries) / len(track_summaries), 2)
        if track_summaries
        else 0.0
    )
    avg_track_duration_sec = round(avg_track_duration_frames / fps, 2) if fps > 0 else 0.0
    longest_track_duration_frames = int(track_summaries[0]["duration_frames"]) if track_summaries else 0
    longest_track_duration_sec = round(longest_track_duration_frames / fps, 2) if fps > 0 else 0.0

    report: dict[str, Any] = {
        "video_path": str(video_path),
        "output_video_path": str(output_video_path),
        "total_frames": frame_index,
        "fps": round(float(fps), 3),
        "duration_sec": round((frame_index / fps) if frame_index else 0.0, 2),
        "confidence_threshold": conf_threshold,
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "detection_frame_ratio": round((frames_with_detections / frame_index) if frame_index else 0.0, 4),
        "max_detections_in_frame": int(max(per_frame_counts, default=0)),
        "avg_detections_per_frame": round((total_detections / frame_index) if frame_index else 0.0, 4),
        "class_counts": dict(class_counts),
        "timeline_detections_per_second": {
            str(second): count for second, count in sorted(timeline_counts.items())
        },
        "top_confidence_events": top_events,
        "vlm_candidate_crops": saved_candidates,
        "tracking": {
            "enabled": bool(tracking),
            "tracker": tracker if tracking else None,
            "tracked_detections": tracked_detection_count,
            "unique_tracks": len(track_summaries),
            "avg_track_duration_frames": avg_track_duration_frames,
            "avg_track_duration_sec": avg_track_duration_sec,
            "longest_track_duration_frames": longest_track_duration_frames,
            "longest_track_duration_sec": longest_track_duration_sec,
            "top_tracks": track_summaries[:20],
        },
    }

    return report


def encode_image_to_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def analyze_single_crop_with_groq(client: Any, vlm_model: str, image_path: Path) -> tuple[dict[str, Any], str]:
    data_url = encode_image_to_data_url(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DRONE_ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=vlm_model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(
            model=vlm_model,
            messages=messages,
            temperature=0,
        )

    content: Any = completion.choices[0].message.content if completion.choices else ""
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    raw_text = str(content or "")

    parsed = parse_json_payload(raw_text)
    normalized = normalize_vlm_analysis(parsed)
    return normalized, raw_text


def analyze_crops_with_vlm(
    candidates: list[dict[str, Any]],
    vlm_provider: str,
    vlm_model: str,
    groq_api_key: Optional[str],
) -> dict[str, Any]:
    provider = vlm_provider.lower()

    if provider == "none":
        return {
            "status": "disabled",
            "provider": "none",
            "model": vlm_model,
            "reason": "VLM disabled by --vlm-provider none.",
            "results": [],
        }

    if not candidates:
        return {
            "status": "skipped",
            "provider": provider,
            "model": vlm_model,
            "reason": "No candidate crops were available for VLM analysis.",
            "results": [],
        }

    if provider != "groq":
        return {
            "status": "error",
            "provider": provider,
            "model": vlm_model,
            "reason": "Only 'groq' VLM provider is currently supported.",
            "results": [],
        }

    if Groq is None:
        return {
            "status": "error",
            "provider": provider,
            "model": vlm_model,
            "reason": "groq package is not installed. Install dependencies again.",
            "results": [],
        }

    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "provider": provider,
            "model": vlm_model,
            "reason": "Missing Groq API key. Set GROQ_API_KEY or pass --groq-api-key.",
            "results": [],
        }

    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    client = Groq(api_key=api_key)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates, start=1):
        crop_path = Path(str(candidate.get("crop_path", "")))
        if not crop_path.exists():
            errors.append(
                {
                    "crop_path": str(crop_path),
                    "error": "Crop image file not found.",
                }
            )
            continue

        try:
            vlm_analysis, _raw_response = analyze_single_crop_with_groq(client, vlm_model, crop_path)
            unified_score = compute_unified_score(
                yolo_confidence=safe_float(candidate.get("yolo_confidence"), 0.0),
                threat_level=str(vlm_analysis.get("threat_level", "none")),
                vlm_confidence=safe_float(vlm_analysis.get("confidence"), 0.0),
            )

            results.append(
                {
                    "rank": idx,
                    "frame": candidate.get("frame"),
                    "time_sec": candidate.get("time_sec"),
                    "label": candidate.get("label"),
                    "track_id": candidate.get("track_id"),
                    "bbox_xyxy": candidate.get("bbox_xyxy"),
                    "crop_path": str(crop_path),
                    "yolo_confidence": candidate.get("yolo_confidence"),
                    "vlm_analysis": vlm_analysis,
                    "unified_score": unified_score,
                    "unified_band": score_to_risk_band(unified_score),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "crop_path": str(crop_path),
                    "error": str(exc),
                }
            )

    threat_counts = Counter(item["vlm_analysis"]["threat_level"] for item in results)
    max_unified = max((safe_float(item.get("unified_score"), 0.0) for item in results), default=0.0)
    avg_unified = (
        round(sum(safe_float(item.get("unified_score"), 0.0) for item in results) / len(results), 4)
        if results
        else 0.0
    )

    top_risk = max(results, key=lambda item: safe_float(item.get("unified_score"), 0.0), default=None)
    top_risk_summary: Optional[dict[str, Any]] = None
    if top_risk is not None:
        top_risk_summary = {
            "frame": top_risk.get("frame"),
            "time_sec": top_risk.get("time_sec"),
            "track_id": top_risk.get("track_id"),
            "crop_path": top_risk.get("crop_path"),
            "yolo_confidence": top_risk.get("yolo_confidence"),
            "threat_level": top_risk.get("vlm_analysis", {}).get("threat_level"),
            "vlm_confidence": top_risk.get("vlm_analysis", {}).get("confidence"),
            "unified_score": top_risk.get("unified_score"),
            "unified_band": top_risk.get("unified_band"),
            "payload_detected": top_risk.get("vlm_analysis", {}).get("payload_detected"),
            "payload_type": top_risk.get("vlm_analysis", {}).get("payload_type"),
            "threat_reasoning": top_risk.get("vlm_analysis", {}).get("threat_reasoning"),
        }

    return {
        "status": "ok" if results else "error",
        "provider": provider,
        "model": vlm_model,
        "prompt": DRONE_ANALYSIS_PROMPT,
        "unified_score_formula": "0.55 * yolo_confidence + 0.45 * (threat_level_score * vlm_confidence)",
        "threat_level_score_map": THREAT_LEVEL_VALUES,
        "analyzed_crops": len(results),
        "avg_unified_score": avg_unified,
        "max_unified_score": round(max_unified, 4),
        "threat_level_counts": dict(threat_counts),
        "top_risk_detection": top_risk_summary,
        "results": results,
        "errors": errors,
    }


def vlm_analysis_to_context_text(vlm_analysis: dict[str, Any]) -> str:
    status = str(vlm_analysis.get("status", "unknown"))
    lines = [f"VLM status: {status}"]

    reason = vlm_analysis.get("reason")
    if reason:
        lines.append(f"VLM reason: {reason}")

    if status != "ok":
        return "\n".join(lines)

    lines.extend(
        [
            f"VLM provider: {vlm_analysis.get('provider')}",
            f"VLM model: {vlm_analysis.get('model')}",
            f"Analyzed crops: {vlm_analysis.get('analyzed_crops')}",
            f"Average unified score: {vlm_analysis.get('avg_unified_score')}",
            f"Max unified score: {vlm_analysis.get('max_unified_score')}",
            f"Threat counts: {vlm_analysis.get('threat_level_counts')}",
        ]
    )

    top_risk = vlm_analysis.get("top_risk_detection")
    if isinstance(top_risk, dict):
        lines.append(f"Top risk detection: {top_risk}")

    top_items = sorted(
        vlm_analysis.get("results", []),
        key=lambda item: safe_float(item.get("unified_score"), 0.0),
        reverse=True,
    )[:5]

    if top_items:
        lines.append("Top unified score items:")
        for item in top_items:
            lines.append(
                "- frame={frame}, time={time_sec}, yolo_conf={yolo_conf}, threat={threat}, "
                "vlm_conf={vlm_conf}, unified={unified}".format(
                    frame=item.get("frame"),
                    time_sec=item.get("time_sec"),
                    yolo_conf=item.get("yolo_confidence"),
                    threat=item.get("vlm_analysis", {}).get("threat_level"),
                    vlm_conf=item.get("vlm_analysis", {}).get("confidence"),
                    unified=item.get("unified_score"),
                )
            )

    return "\n".join(lines)


def report_to_context_text(report: dict[str, Any]) -> str:
    tracking = report.get("tracking", {})

    lines = [
        f"Video path: {report['video_path']}",
        f"Output video path: {report['output_video_path']}",
        f"Total frames: {report['total_frames']}",
        f"Duration (sec): {report['duration_sec']}",
        f"Total detections: {report['total_detections']}",
        f"Frames with detections: {report['frames_with_detections']}",
        f"Detection frame ratio: {report['detection_frame_ratio']}",
        f"Max detections in one frame: {report['max_detections_in_frame']}",
        f"Average detections per frame: {report['avg_detections_per_frame']}",
        f"Class counts: {report['class_counts']}",
        f"VLM candidate crops: {len(report.get('vlm_candidate_crops', []))}",
    ]

    if isinstance(tracking, dict) and tracking.get("enabled"):
        lines.extend(
            [
                "Tracking enabled: True",
                f"Unique tracked drones: {tracking.get('unique_tracks', 0)}",
                f"Tracked detections: {tracking.get('tracked_detections', 0)}",
                f"Average track duration (sec): {tracking.get('avg_track_duration_sec', 0)}",
                f"Longest track duration (sec): {tracking.get('longest_track_duration_sec', 0)}",
            ]
        )

        top_tracks = tracking.get("top_tracks", [])
        if isinstance(top_tracks, list) and top_tracks:
            lines.append("Top tracks:")
            for track in top_tracks[:5]:
                if not isinstance(track, dict):
                    continue
                lines.append(
                    "- id={track_id}, label={label}, duration_sec={duration_sec}, max_conf={max_conf}".format(
                        track_id=track.get("track_id"),
                        label=track.get("label"),
                        duration_sec=track.get("duration_sec"),
                        max_conf=track.get("max_confidence"),
                    )
                )
    else:
        lines.append("Tracking enabled: False")

    timeline = report.get("timeline_detections_per_second", {})
    busiest_seconds = sorted(
        ((int(second), int(count)) for second, count in timeline.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    if busiest_seconds:
        lines.append("Busiest seconds (second -> detections):")
        lines.extend([f"- {second}: {count}" for second, count in busiest_seconds])

    lines.append(f"Top confidence events: {report.get('top_confidence_events', [])}")

    vlm_analysis = report.get("vlm_analysis", {})
    if isinstance(vlm_analysis, dict):
        lines.append("\nVLM threat analysis:")
        lines.append(vlm_analysis_to_context_text(vlm_analysis))

    return "\n".join(lines)


def fallback_summary(report: dict[str, Any]) -> str:
    total_frames = report.get("total_frames", 0)
    duration_sec = report.get("duration_sec", 0)
    total_detections = report.get("total_detections", 0)
    frames_with_detections = report.get("frames_with_detections", 0)
    class_counts: dict[str, int] = report.get("class_counts", {})

    if not class_counts:
        return (
            f"Processed {total_frames} frames ({duration_sec}s). "
            "No detections were found at the selected confidence threshold."
        )

    top_class, top_count = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[0]
    ratio_pct = report.get("detection_frame_ratio", 0.0) * 100.0

    base_summary = (
        f"Processed {total_frames} frames ({duration_sec}s), found {total_detections} detections in "
        f"{frames_with_detections} frames ({ratio_pct:.1f}%). Most frequent class: {top_class} ({top_count})."
    )

    tracking = report.get("tracking", {})
    if isinstance(tracking, dict) and tracking.get("enabled"):
        unique_tracks = int(tracking.get("unique_tracks", 0))
        base_summary = f"{base_summary} Tracking identified {unique_tracks} unique drone tracks."

    vlm_analysis = report.get("vlm_analysis", {})
    if isinstance(vlm_analysis, dict) and vlm_analysis.get("status") == "ok":
        return (
            f"{base_summary} VLM analyzed {vlm_analysis.get('analyzed_crops')} crops with average unified "
            f"score {vlm_analysis.get('avg_unified_score')} and max unified score "
            f"{vlm_analysis.get('max_unified_score')}."
        )

    return base_summary


def markdown_cell(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).replace("|", "/").replace("\n", " ").strip()
    return text if text else "-"


def summarize_counter(counter: Counter[str]) -> str:
    if not counter:
        return "-"
    parts = [f"{key}: {count}" for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
    return ", ".join(parts)


def safe_bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def fallback_final_report(report: dict[str, Any]) -> str:
    vlm_analysis = report.get("vlm_analysis", {})
    tracking = report.get("tracking", {})
    results: list[dict[str, Any]] = []

    if isinstance(vlm_analysis, dict):
        raw_results = vlm_analysis.get("results", [])
        if isinstance(raw_results, list):
            results = [item for item in raw_results if isinstance(item, dict)]

    results = sorted(results, key=lambda item: safe_float(item.get("unified_score"), 0.0), reverse=True)
    top_risk = results[0] if results else None

    drone_type_counter: Counter[str] = Counter()
    payload_type_counter: Counter[str] = Counter()
    size_counter: Counter[str] = Counter()
    threat_level_counter: Counter[str] = Counter()
    payload_detected_count = 0

    for item in results:
        analysis = item.get("vlm_analysis", {})
        if not isinstance(analysis, dict):
            continue

        drone_type_counter[str(analysis.get("drone_type", "unknown"))] += 1
        payload_type_counter[str(analysis.get("payload_type", "unknown"))] += 1
        size_counter[str(analysis.get("estimated_size", "unknown"))] += 1
        threat_level_counter[str(analysis.get("threat_level", "unknown"))] += 1
        if bool(analysis.get("payload_detected", False)):
            payload_detected_count += 1

    total_detections = int(report.get("total_detections", 0))
    frames_with_detections = int(report.get("frames_with_detections", 0))
    duration_sec = safe_float(report.get("duration_sec", 0.0), 0.0)
    unique_tracked = (
        int(tracking.get("unique_tracks", 0))
        if isinstance(tracking, dict) and tracking.get("enabled")
        else 0
    )

    lines = [
        "# Drone Activity Report",
        "",
        "## Quick Summary",
        f"- Drones detected: {total_detections}",
        f"- Time monitored: {duration_sec:.2f} seconds",
        f"- Frames containing drones: {frames_with_detections}",
        f"- Unique drones tracked: {unique_tracked}",
    ]

    highest_threat = "none"
    if threat_level_counter:
        for level in ["critical", "high", "medium", "low", "none"]:
            if threat_level_counter.get(level, 0) > 0:
                highest_threat = level
                break

    if highest_threat == "critical":
        overall_risk = "Critical"
    elif highest_threat == "high":
        overall_risk = "High"
    elif highest_threat == "medium":
        overall_risk = "Medium"
    elif highest_threat == "low":
        overall_risk = "Low"
    else:
        overall_risk = "None"

    lines.append(f"- Overall risk level: {overall_risk}")

    if isinstance(top_risk, dict):
        lines.append(f"- Highest risk category observed: {markdown_cell(top_risk.get('unified_band')).title()}")
    if results:
        lines.append(f"- Possible payload detected in {payload_detected_count} out of {len(results)} reviewed drone views")

    lines.extend(["", "## Main Findings"])
    lines.append(f"- Most common drone type: {summarize_counter(drone_type_counter)}")
    lines.append(f"- Most common payload type: {summarize_counter(payload_type_counter)}")
    lines.append(f"- Common drone size: {summarize_counter(size_counter)}")
    lines.append(f"- Threat distribution: {summarize_counter(threat_level_counter)}")

    lines.extend(["", "## Tracked Drones"])
    if isinstance(tracking, dict) and tracking.get("enabled"):
        top_tracks = tracking.get("top_tracks", [])
        if isinstance(top_tracks, list) and top_tracks:
            lines.extend(
                [
                    "| Drone ID | Seen For (sec) | Appearances | Confidence Peak |",
                    "|---:|---:|---:|---:|",
                ]
            )

            for track in top_tracks[:10]:
                if not isinstance(track, dict):
                    continue
                lines.append(
                    "| {track_id} | {duration_sec} | {appearances} | {max_conf} |".format(
                        track_id=markdown_cell(track.get("track_id")),
                        duration_sec=markdown_cell(track.get("duration_sec")),
                        appearances=markdown_cell(track.get("detection_count")),
                        max_conf=markdown_cell(track.get("max_confidence")),
                    )
                )
        else:
            lines.append("- Tracking was enabled, but no stable tracks were found.")
    else:
        lines.append("- Tracking is disabled for this run.")

    lines.extend(["", "## Drone Assessments (Top 5)"])
    if results:
        lines.extend(
            [
                "| # | Drone ID | Drone Type | Payload | Size | Threat | Reason | Features |",
                "|---:|---:|---|---|---|---|---|---|",
            ]
        )

        for idx, item in enumerate(results[:5], start=1):
            analysis = item.get("vlm_analysis", {})
            if not isinstance(analysis, dict):
                analysis = {}

            payload_type = markdown_cell(analysis.get("payload_type"))
            payload_desc = analysis.get("payload_description")
            if bool(analysis.get("payload_detected", False)):
                payload_text = payload_type if payload_desc in {None, "", "-"} else f"{payload_type}: {payload_desc}"
            else:
                payload_text = "No visible payload"

            notable_features = analysis.get("notable_features", [])
            if isinstance(notable_features, list):
                features_text = ", ".join(str(feature).strip() for feature in notable_features if str(feature).strip())
            else:
                features_text = str(notable_features or "")

            lines.append(
                "| {rank} | {track_id} | {drone_type} | {payload} | {size} | {threat} | {reason} | {features} |".format(
                    rank=idx,
                    track_id=markdown_cell(item.get("track_id")),
                    drone_type=markdown_cell(analysis.get("drone_type")),
                    payload=markdown_cell(payload_text),
                    size=markdown_cell(analysis.get("estimated_size")),
                    threat=markdown_cell(analysis.get("threat_level")),
                    reason=markdown_cell(analysis.get("threat_reasoning")),
                    features=markdown_cell(features_text),
                )
            )
    else:
        lines.append("- No detailed drone assessments were available.")

    lines.extend(
        [
            "",
            "## Recommended Next Actions",
        ]
    )

    if highest_threat in {"critical", "high"}:
        lines.extend(
            [
                "- Escalate immediately and trigger incident response workflow.",
                "- Verify payload evidence from top-risk crops using a human analyst.",
                "- Cross-check with additional sensors/cameras before engagement.",
            ]
        )
    elif highest_threat == "medium":
        lines.extend(
            [
                "- Increase monitoring frequency in the detected area.",
                "- Re-run with stricter thresholds to reduce uncertain detections.",
                "- Review top unified-score crops with a human operator.",
            ]
        )
    else:
        lines.extend(
            [
                "- Continue normal monitoring.",
                "- Re-check if the environment or mission context changes.",
                "- Keep collecting examples to improve future detection quality.",
            ]
        )

    return "\n".join(lines)


def build_langchain_chain(llm_provider: str, llm_model: str, groq_api_key: Optional[str]) -> Any:
    provider = llm_provider.lower()
    print(provider , llm_model)
    if provider == "none" or llm_model.lower() == "none":
        return None
    print("huuu")
    prompt = ChatPromptTemplate.from_template(
        """
You are a drone detection and threat-assessment assistant.
Use ONLY the report context below to answer.
If the answer is not in the report, say you do not have enough data.

Report context:
{report}

User question:
{question}

Answer with concise, factual details.
""".strip()
    )

    if provider == "ollama":
        if ChatOllama is None:
            print("hiiiii ollama")
            return None
        llm = ChatOllama(
            model=llm_model,
            temperature=0.2
        )
        print("LLM ollama" , llm)
    elif provider == "groq":
        if ChatGroq is None:
            print("hiiiii groq")
            return None

        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

        if not os.getenv("GROQ_API_KEY"):
            return None

        llm = ChatGroq(model=llm_model, temperature=0)
        print("LLm groq" , llm)
    else:
        return None

    return prompt | llm | StrOutputParser()


def generate_summary(chain: Any, report_text: str, report: dict[str, Any]) -> tuple[str, bool]:
    if chain is None:
        return fallback_summary(report), False

    try:
        summary = chain.invoke(
            {
                "report": report_text,
                "question": (
                    "Summarize detections and VLM threat findings, including unified score highlights."
                ),
            }
        )
        return str(summary).strip(), True
    except Exception as exc:
        fallback = fallback_summary(report)
        return f"{fallback}\n\nLangChain summary unavailable: {exc}", False


def generate_final_report(chain: Any, report_text: str, report: dict[str, Any]) -> tuple[str, bool]:
    structured_report = fallback_final_report(report)

    if chain is None:
        return structured_report, False

    try:
        user_narrative = chain.invoke(
            {
                "report": report_text,
                "question": (
                    "Write a short plain-language summary (4-6 sentences) for everyday users. "
                    "Explain what was detected, the general risk level, and what action to take next. "
                    "Avoid model names, file paths, formulas, and technical terms."
                ),
            }
        )
        user_narrative_text = str(user_narrative).strip()
        if user_narrative_text:
            final_report = (
                f"{structured_report}\n\n"
                "## Plain-Language Summary\n"
                f"{user_narrative_text}"
            )
            return final_report, True
        return structured_report, False
    except Exception:
        return structured_report, False


def run_chat(chain: Any, report_text: str) -> None:
    print("\nChat mode is ready. Ask about detections or threat analysis. Type 'exit' to quit.")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Chat ended.")
            break

        try:
            answer = chain.invoke({"report": report_text, "question": question})
            print(f"Assistant: {answer}\n")
        except Exception as exc:
            print(f"Assistant: Could not answer with LangChain provider: {exc}\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO drone detection, crop detected regions, analyze with Groq VLM, and generate an LLM report."
        )
    )
    # Check if weights/best.pt exists locally, use that; otherwise default to yolo26n.pt
    default_weights = "weights/model.pt" if (Path.cwd() / "weights" / "model.pt").exists() else "yolo26n.pt"
    parser.add_argument(
        "--weights",
        default=default_weights,
        help="Path to local .pt/.onnx weights or YOLO model alias. Defaults to weights/model.pt if it exists, otherwise yolo26n.pt.",
    )
    parser.add_argument("--video", default="drone.mp4", help="Path to input video")
    parser.add_argument(
        "--output-video",
        default="outputs/drone_annotated.mp4",
        help="Path to save annotated output video",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/detection_report.json",
        help="Path to save full JSON report",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/final_threat_report.md",
        help="Path to save final markdown report",
    )
    parser.add_argument(
        "--crop-dir",
        default="outputs/crops",
        help="Directory where YOLO crop images are saved for VLM analysis",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cpu', '0', or '0,1'. Leave empty for auto.",
    )
    parser.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable tracking across frames (default: enabled). Use --no-tracking to disable.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config when tracking is enabled, for example bytetrack.yaml or botsort.yaml.",
    )
    parser.add_argument(
        "--vlm-provider",
        default="groq",
        choices=["groq", "none"],
        help="VLM backend for crop analysis.",
    )
    parser.add_argument(
        "--vlm-model",
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        help="Groq vision model for image threat analysis.",
    )
    parser.add_argument(
        "--vlm-max-crops",
        type=int,
        default=12,
        help="Maximum number of crop images to send to VLM.",
    )
    parser.add_argument(
        "--vlm-min-yolo-conf",
        type=float,
        default=0.35,
        help="Minimum YOLO confidence to keep a crop for VLM analysis.",
    )
    parser.add_argument(
        "--llm-provider",
        default="ollama",
        choices=["ollama", "groq", "none"],
        help="LLM backend for summary, final report, and chat.",
    )
    parser.add_argument(
        "--llm-model",
        default="deepseek-r1:1.5b",
        choices=["qwen2.5:3b","deepseek-r1:1.5b","llama-3.3-70b-versatile",None],
        help="Model name for selected LLM provider. Defaults to a provider-appropriate model.",
    )
    parser.add_argument(
        "--groq-api-key",
        default=None,
        help="Optional Groq API key. If omitted, reads GROQ_API_KEY from environment.",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Run inference and report generation only, without interactive chat.",
    )
    return parser.parse_args()

def serialize_paths(obj):
    from pathlib import Path

    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_paths(i) for i in obj]
    else:
        return obj

def run_full_pipeline(video_path,  threshold_confidence =0.35 , top_K = 12 , llmProvider = "Groq" ,  root_dir="./uploads" ):
    video_path = Path(video_path)
    root_dir = Path(root_dir)

    args = parse_args()
    args.vlm_min_yolo_conf = threshold_confidence
    args.conf=threshold_confidence
    args.vlm_max_crops = top_K
    args.llm_provider = llmProvider
    root_dir = Path(__file__).resolve().parent

    weights_source = resolve_model_source(args.weights, root_dir)
    #video_path = resolve_path(args.video, root_dir)
    output_video_path = resolve_path(args.output_video, root_dir)
    output_json_path = resolve_path(args.output_json, root_dir)
    output_report_path = resolve_path(args.output_report, root_dir)
    crop_output_dir = resolve_path(args.crop_dir, root_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    
    if args.llm_provider.lower() == "groq":
        llm_model = "llama-3.3-70b-versatile"
    elif args.llm_provider.lower() == "ollama":
        llm_model = "deepseek-r1:1.5b"
    else:
        llm_model = "none"

    print(f"Loading model: {weights_source}")
    model = YOLO(weights_source)

    report = run_video_inference(
        model=model,
        video_path=video_path,
        output_video_path=output_video_path,
        conf_threshold=args.conf,
        device=args.device,
        crop_output_dir=crop_output_dir,
        vlm_min_yolo_conf=args.vlm_min_yolo_conf,
        vlm_max_crops=args.vlm_max_crops,
        tracking=args.tracking,
        tracker=args.tracker,
    )
    vlm_analysis = analyze_crops_with_vlm(
        candidates=report.get("vlm_candidate_crops", []),
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        groq_api_key=args.groq_api_key,
    )
    report["vlm_analysis"] = vlm_analysis
    report_text = report_to_context_text(report)
    chain = build_langchain_chain(args.llm_provider, llm_model, args.groq_api_key)
    summary, _summary_ok = generate_summary(chain, report_text, report)
    print(type(summary), summary)
    final_report, final_report_from_llm = generate_final_report(chain, report_text, report)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(final_report, encoding="utf-8")   
    retrn = {
        "report_path": str(output_report_path),
        "report_text": final_report,
        "report_from": "LLM" if final_report_from_llm else "fallback",
        "raw_report": report,
        "summary": summary,
        "summary_from": "LLM" if _summary_ok else "fallback"
    }
    return [retrn , chain]

def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    weights_source = resolve_model_source(args.weights, root_dir)
    #video_path = resolve_path(args.video, root_dir)
    video_path = resolve_path(args.video, root_dir)  # Hardcoded for testing; change back to args.video for production
    output_video_path = resolve_path(args.output_video, root_dir)
    output_json_path = resolve_path(args.output_json, root_dir)
    output_report_path = resolve_path(args.output_report, root_dir)
    crop_output_dir = resolve_path(args.crop_dir, root_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    llm_model = args.llm_model
    if not llm_model:
        if args.llm_provider == "groq":
            llm_model = "llama-3.3-70b-versatile"
        elif args.llm_provider == "ollama":
            llm_model = "llama3.1"
        else:
            llm_model = "none"

    print(f"Loading model: {weights_source}")
    model = YOLO(weights_source)

    report = run_video_inference(
        model=model,
        video_path=video_path,
        output_video_path=output_video_path,
        conf_threshold=args.conf,
        device=args.device,
        crop_output_dir=crop_output_dir,
        vlm_min_yolo_conf=args.vlm_min_yolo_conf,
        vlm_max_crops=args.vlm_max_crops,
        tracking=args.tracking,
        tracker=args.tracker,
    )

    vlm_analysis = analyze_crops_with_vlm(
        candidates=report.get("vlm_candidate_crops", []),
        vlm_provider=args.vlm_provider,
        vlm_model=args.vlm_model,
        groq_api_key=args.groq_api_key,
    )
    report["vlm_analysis"] = vlm_analysis

    report_text = report_to_context_text(report)
    chain = build_langchain_chain(args.llm_provider, llm_model, args.groq_api_key)
    summary, _summary_ok = generate_summary(chain, report_text, report)
    final_report, final_report_from_llm = generate_final_report(chain, report_text, report)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(final_report, encoding="utf-8")

    print("\n=== Detection + Threat Summary ===")
    print(summary)
    print(f"\nAnnotated video saved to: {output_video_path}")
    print(f"Detection report saved to: {output_json_path}")
    print(f"Final threat report saved to: {output_report_path}")
    print(f"Final report source: {'LLM' if final_report_from_llm else 'fallback template'}")

    if not args.skip_chat:
        if chain is not None:
            run_chat(chain, report_text)
        else:
            print("\nLangChain chat disabled because no working LLM backend is available.")
            if args.llm_provider == "groq":
                print("Set GROQ_API_KEY or pass --groq-api-key, and use a valid Groq text model.")
            elif args.llm_provider == "ollama":
                print("Install and run Ollama, then set --llm-model to a local model name.")
            else:
                print("Set --llm-provider to ollama or groq to enable interactive chat.")


if __name__ == "__main__":
    main()
