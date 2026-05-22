"""
Real-time Detection Flask Server with WebSocket & RAG Support
Requires: pip install flask flask-cors flask-socketio python-socketio python-engineio faiss-cpu
"""

import os
import json
import threading
import queue
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: faiss-cpu not installed. RAG features will be disabled.")
    print("Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

# =============================
# INITIALIZATION
# =============================

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120, ping_interval=25)

UPLOAD_FOLDER = "uploads"
SESSIONS_FOLDER = "sessions"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSIONS_FOLDER, exist_ok=True)

# Load YOLO model once
try:
    from ultralytics import YOLO
    model = YOLO("model.pt")
    print("✅ YOLO model loaded")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    model = None

# =============================
# RAG SYSTEM (Optional)
# =============================

class SimpleRAGSystem:
    """Lightweight RAG without faiss for fallback"""
    
    def __init__(self):
        self.knowledge_base = []
        self.lock = threading.Lock()
    
    def add_detection(self, detection: Dict[str, Any]):
        """Add detection to knowledge base"""
        with self.lock:
            self.knowledge_base.append({
                'timestamp': datetime.now().isoformat(),
                'detection': detection
            })
    
    def search_similar(self, query: Dict[str, Any], top_k: int = 3) -> List[Dict]:
        """Simple string-based search"""
        query_label = str(query.get('label', '')).lower()
        results = []
        
        with self.lock:
            for item in self.knowledge_base:
                if query_label in str(item['detection'].get('label', '')).lower():
                    results.append(item)
        
        return results[-top_k:] if results else []

class FaissRAGSystem:
    """RAG system using FAISS for similarity search"""
    
    def __init__(self):
        self.index = None
        self.metadata = []
        self.embeddings = []
        self.lock = threading.Lock()
        self.dimension = 128  # Feature vector dimension
    
    def add_detection(self, detection: Dict[str, Any]):
        """Add detection with embedding to RAG"""
        try:
            # Simple feature vector from detection data
            features = self._extract_features(detection)
            
            with self.lock:
                if self.index is None:
                    self.index = faiss.IndexFlatL2(self.dimension)
                
                self.metadata.append({
                    'timestamp': datetime.now().isoformat(),
                    'detection': detection
                })
                self.embeddings.append(features)
                
                # Add to index
                embedding_array = np.array([features], dtype=np.float32)
                self.index.add(embedding_array)
        except Exception as e:
            print(f"❌ RAG add_detection error: {e}")
    
    def search_similar(self, query: Dict[str, Any], top_k: int = 3) -> List[Dict]:
        """Find similar detections"""
        try:
            if self.index is None or len(self.metadata) == 0:
                return []
            
            query_features = self._extract_features(query)
            query_array = np.array([query_features], dtype=np.float32)
            
            with self.lock:
                distances, indices = self.index.search(query_array, min(top_k, len(self.metadata)))
            
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.metadata):
                    results.append(self.metadata[idx])
            
            return results
        except Exception as e:
            print(f"❌ RAG search error: {e}")
            return []
    
    def _extract_features(self, detection: Dict) -> List[float]:
        """Extract feature vector from detection"""
        features = [0.0] * self.dimension
        
        # Confidence (0-1) -> scale to 0-100
        features[0] = detection.get('confidence', 0) * 100
        features[1] = detection.get('yolo_confidence', 0) * 100
        
        # Label encoding (simple hash)
        label = str(detection.get('label', 'unknown')).lower()
        label_hash = hash(label) % 100
        features[2] = label_hash
        
        # Threat level encoding
        threat_level = str(detection.get('threat_level', 'none')).lower()
        threat_map = {'none': 0, 'low': 25, 'medium': 50, 'high': 75, 'critical': 100}
        features[3] = threat_map.get(threat_level, 0)
        
        # Payload detection
        features[4] = 1.0 if detection.get('payload_detected') else 0.0
        
        # Fill rest with detection count patterns
        for i in range(5, self.dimension):
            features[i] = (i * label_hash) % 100
        
        return features

# Initialize RAG system
try:
    if FAISS_AVAILABLE:
        print("✅ Initializing FAISS RAG system...")
        rag_system = FaissRAGSystem()
    else:
        print("✅ Initializing fallback SimpleRAG system...")
        rag_system = SimpleRAGSystem()
except Exception as e:
    print(f"⚠️ RAG initialization failed: {e}. Using SimpleRAG.")
    rag_system = SimpleRAGSystem()

# =============================
# SESSION MANAGEMENT
# =============================

class RealtimeSession:
    """Manages a single real-time detection session"""
    
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.created_at = datetime.now()
        self.status = 'initializing'  # initializing, running, paused, stopped
        self.config = {}
        self.metrics = {
            'frame_count': 0,
            'detection_count': 0,
            'total_detections': 0,
            'frames_with_detections': 0,
            'class_counts': {},
            'timeline_detections_per_second': {},
            'timeline_detections_per_second_per_class': {},
            'top_confidence_events': [],
            'vlm_results': [],
        }
        self.detections_queue = queue.Queue()
        self.report = None
        self.lock = threading.Lock()
    
    def update_metrics(self, detection_data: Dict[str, Any]):
        """Update session metrics"""
        with self.lock:
            self.metrics['detection_count'] += 1
            self.metrics['total_detections'] = self.metrics.get('total_detections', 0) + 1
            
            label = detection_data.get('label', 'unknown')
            self.metrics['class_counts'][label] = self.metrics['class_counts'].get(label, 0) + 1
            
            # Add to top events
            event = {
                'frame': detection_data.get('frame'),
                'time_sec': detection_data.get('time_sec'),
                'label': label,
                'confidence': detection_data.get('confidence'),
            }
            self.metrics['top_confidence_events'].append(event)
            self.metrics['top_confidence_events'] = self.metrics['top_confidence_events'][-50:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        with self.lock:
            return {
                'user_id': self.user_id,
                'session_id': self.session_id,
                'created_at': self.created_at.isoformat(),
                'status': self.status,
                'config': self.config,
                'metrics': self.metrics,
            }

# Active sessions storage
sessions: Dict[str, RealtimeSession] = {}
sessions_lock = threading.Lock()

# =============================
# WEBSOCKET HANDLERS
# =============================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    user_id = request.args.get('user_id')
    print(f"✅ Client connected: {user_id}")
    
    if user_id:
        join_room(f"user_{user_id}")
    
    emit('connection_response', {
        'status': 'connected',
        'message': f'Connected with user_id: {user_id}'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"⚠️ Client disconnected")

@socketio.on('start_realtime_detection')
def handle_start_detection(data):
    """Start real-time detection session"""
    try:
        user_id = data.get('user_id')
        if not user_id:
            emit('error', {'message': 'No user_id provided'})
            return False
        
        session_id = str(uuid.uuid4())
        
        # Create new session
        session = RealtimeSession(user_id, session_id)
        session.config = {
            'video_source': data.get('video_source', '0'),
            'conf': data.get('conf', 0.25),
            'vlm_provider': data.get('vlm_provider', 'cosmos'),
            'llm_provider': data.get('llm_provider', 'groq'),
        }
        session.status = 'running'
        
        with sessions_lock:
            sessions[session_id] = session
        
        print(f"✅ Detection started. Session: {session_id}, User: {user_id}")
        
        # Start detection worker thread
        worker_thread = threading.Thread(
            target=run_detection_worker,
            args=(session_id,),
            daemon=True
        )
        worker_thread.start()
        
        emit('session_started', {
            'success': True,
            'session_id': session_id,
            'user_id': user_id,
        }, room=f"user_{user_id}")
        
        return True
    
    except Exception as e:
        print(f"❌ Start detection error: {e}")
        emit('error', {'message': f'Failed to start detection: {str(e)}'})
        return False

@socketio.on('pause_detection')
def handle_pause_detection(data):
    """Pause detection"""
    session_id = data.get('session_id')
    
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id].status = 'paused'
            print(f"⏸️  Session {session_id} paused")
            emit('pause_confirmed', {'success': True})
            return True
    
    emit('error', {'message': 'Session not found'})
    return False

@socketio.on('resume_detection')
def handle_resume_detection(data):
    """Resume detection"""
    session_id = data.get('session_id')
    
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id].status = 'running'
            print(f"▶️  Session {session_id} resumed")
            emit('resume_confirmed', {'success': True})
            return True
    
    emit('error', {'message': 'Session not found'})
    return False

@socketio.on('stop_detection')
def handle_stop_detection(data):
    """Stop detection and return report"""
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    
    with sessions_lock:
        if session_id not in sessions:
            emit('error', {'message': 'Session not found'})
            return False
        
        session = sessions[session_id]
        session.status = 'stopped'
        
        # Generate report
        report = {
            'session_id': session_id,
            'user_id': user_id,
            'config': session.config,
            'metrics': session.metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        session.report = report
        
        # Save to file
        report_path = Path(SESSIONS_FOLDER) / f"{session_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Session {session_id} stopped. Report saved.")
        
        emit('detection_stopped', {
            'success': True,
            'session_id': session_id,
            'report': report,
        }, room=f"user_{user_id}")
        
        return True

@socketio.on('get_session_status')
def handle_get_status(data):
    """Get current session status"""
    session_id = data.get('session_id')
    
    with sessions_lock:
        if session_id in sessions:
            session = sessions[session_id]
            emit('session_status_response', session.to_dict())
            return True
    
    emit('error', {'message': 'Session not found'})
    return False

# =============================
# HTTP API ENDPOINTS
# =============================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Real-time Drone Detection Server',
        'faiss_available': FAISS_AVAILABLE,
    })

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    user_id = request.args.get('user_id')
    
    with sessions_lock:
        if user_id:
            user_sessions = [s for s in sessions.values() if s.user_id == user_id]
        else:
            user_sessions = list(sessions.values())
        
        return jsonify({
            'sessions': [s.to_dict() for s in user_sessions],
            'total': len(user_sessions)
        })

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get specific session details"""
    with sessions_lock:
        if session_id in sessions:
            return jsonify(sessions[session_id].to_dict())
    
    # Try to load from file
    report_path = Path(SESSIONS_FOLDER) / f"{session_id}.json"
    if report_path.exists():
        with open(report_path, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/sessions/<session_id>/report', methods=['GET'])
def get_session_report(session_id):
    """Get detailed session report"""
    report_path = Path(SESSIONS_FOLDER) / f"{session_id}.json"
    
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
        return jsonify(report)
    
    return jsonify({'error': 'Report not found'}), 404

@app.route('/api/sessions/<session_id>/rag-context', methods=['POST'])
def get_rag_context(session_id):
    """Get RAG context for a detection"""
    try:
        detection = request.json
        
        # Search RAG system
        similar = rag_system.search_similar(detection, top_k=3)
        
        return jsonify({
            'success': True,
            'detection': detection,
            'similar_detections': similar,
            'rag_available': FAISS_AVAILABLE,
        })
    except Exception as e:
        print(f"❌ RAG context error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>/threat-analysis', methods=['POST'])
def get_threat_analysis(session_id):
    """Get RAG-enhanced threat analysis"""
    try:
        detection = request.json
        
        # Get similar past detections
        similar = rag_system.search_similar(detection, top_k=5)
        
        # Generate threat reasoning based on similar cases
        threat_level = detection.get('threat_level', 'medium')
        similar_count = len(similar)
        
        analysis = {
            'detection': detection,
            'similar_cases': similar_count,
            'threat_level': threat_level,
            'reasoning': f'Based on {similar_count} similar detection patterns, threat level assessed as {threat_level}.',
            'recommendations': [
                'Continue monitoring',
                'Review crop images',
                'Cross-reference with other sensors' if threat_level in ['high', 'critical'] else '',
            ]
        }
        
        return jsonify(analysis)
    except Exception as e:
        print(f"❌ Threat analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<session_id>/crops', methods=['GET'])
def get_session_crops(session_id):
    """Get VLM-analyzed crops from session"""
    with sessions_lock:
        if session_id in sessions:
            session = sessions[session_id]
            return jsonify({
                'session_id': session_id,
                'crops': session.metrics.get('vlm_results', []),
            })
    
    return jsonify({'error': 'Session not found'}), 404

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system-wide statistics"""
    with sessions_lock:
        total_sessions = len(sessions)
        active_sessions = len([s for s in sessions.values() if s.status == 'running'])
        total_detections = sum(s.metrics.get('total_detections', 0) for s in sessions.values())
    
    return jsonify({
        'total_sessions': total_sessions,
        'active_sessions': active_sessions,
        'total_detections': total_detections,
        'rag_available': FAISS_AVAILABLE,
        'timestamp': datetime.now().isoformat(),
    })

# =============================
# BACKGROUND WORKERS
# =============================

def run_detection_worker(session_id: str):
    """Background worker for real-time detection"""
    try:
        with sessions_lock:
            if session_id not in sessions:
                return
            session = sessions[session_id]
        
        # Simulate detection (replace with actual detection logic)
        import time
        frame_count = 0
        
        while session.status in ['running', 'paused']:
            if session.status == 'paused':
                time.sleep(0.1)
                continue
            
            # Simulate frame processing
            frame_count += 1
            
            # Simulate detection
            if frame_count % 5 == 0:  # Detection every 5 frames
                detection = {
                    'frame': frame_count,
                    'time_sec': frame_count / 30,  # Assume 30 FPS
                    'label': 'drone',
                    'confidence': 0.85,
                    'threat_level': 'medium',
                }
                
                session.update_metrics(detection)
                rag_system.add_detection(detection)
                
                # Emit to client
                socketio.emit('metrics_update', session.metrics, room=f"user_{session.user_id}")
            
            time.sleep(0.033)  # ~30 FPS
    
    except Exception as e:
        print(f"❌ Detection worker error: {e}")

# =============================
# MAIN
# =============================

if __name__ == '__main__':
    print("🚀 Starting Real-time Drone Detection Server...")
    print(f"📡 FAISS/RAG Available: {FAISS_AVAILABLE}")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)
