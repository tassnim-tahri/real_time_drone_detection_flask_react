```markdown
# Unified Real-Time Detection & Analysis Backend API

## Overview

This backend integrates:
- **Real-time YOLO detection** via streaming camera feed
- **Video file analysis** with VLM threat assessment
- **LLM-powered reporting** and chatbot
- **RESTful API** for frontend communication

---

## Architecture

### Components

┌─────────────────┐ │ Live Camera │ └────────┬────────┘ │ ┌────▼─────────┐ ┌──────────────────┐ │ YOLO RT │◄────────│ Confidence │ │ Detection │ │ Threshold │ └────┬─────────┘ └──────────────────┘ │ ┌────▼──────────────┐ │ Video Stream │ │ (MJPEG) │ └────────────────────┘

┌─────────────────┐ │ Video Upload │ └────────┬────────┘ │ ┌────▼─────────────────┐ │ YOLO Inference │ │ + Tracking │ └────┬─────────────────┘ │ ┌────▼──────────────┐ │ Crop Extraction │ └────┬──────────────┘ │ ┌────▼──────────────┐ │ VLM Analysis │ │ (Groq) │ └────┬──────────────┘ │ ┌────▼──────────────┐ │ LLM Report Gen │ │ (Groq) │ └────┬──────────────┘ │ ┌────▼──────────────┐ │ Final Report │ │ + Chat Ready │ └────────────────────┘


---

## API Endpoints

### Streaming & Real-Time Detection

#### `GET /video_feed`
Live MJPEG video stream with real-time detections

**Response:**
- MJPEG stream with bounding boxes and annotations

**Example:**
```html
<img src="http://localhost:8001/video_feed" />





# Real-Time Detection & Analysis Backend

## 📋 Overview

`real_time_flask_app.py` is a unified Flask backend that provides:

✅ **Live camera streaming** with real-time YOLO detection  
✅ **Video file analysis** with background processing  
✅ **Dual dashboard support** - streaming + analysis tabs  
✅ **Confidence threshold control** (live adjustment)  
✅ **Non-blocking analysis** (async video processing)  
✅ **Statistics tracking** (FPS, detections, classes)  

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install flask flask-cors ultralytics opencv-python torch