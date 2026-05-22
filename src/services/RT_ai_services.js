import io from 'socket.io-client';

const SOCKET_SERVER_URL = process.env.REACT_APP_SOCKET_URL || 'http://193.95.31.90:5001';
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://193.95.31.90:5001/api';

let socket = null;
let socketConnected = false;
let currentSessionId = null;
let currentUserId = null;

// Event listeners registry
const eventListeners = {
  frameUpdate: [],
  metricsUpdate: [],
  detectionAlert: [],
  sessionStatus: [],
  error: [],
};

/**
 * Get current user ID from localStorage (set during login)
 */
export function getCurrentUserId() {
  return localStorage.getItem('rt_detection_user_id') || null;
}

/**
 * Initialize socket connection with authenticated user ID
 */
export function initializeSocket() {
  // Check if user is logged in first
  const userId = getCurrentUserId();
  
  if (!userId) {
    console.warn('⚠️ No user_id in localStorage. User must be logged in first.');
    return false;
  }

  if (socket && socketConnected) {
    console.log('✅ Socket already connected');
    return true;
  }

  try {
    currentUserId = userId;
    
    socket = io(SOCKET_SERVER_URL, {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5,
      transports: ['websocket', 'polling'],
      withCredentials: true,
      query: {
        user_id: userId, // ✅ User ID from login
      },
    });

    // Connection events
    socket.on('connect', () => {
      socketConnected = true;
      console.log('✅ Connected to real-time server. User ID:', userId);
      emitEvent('sessionStatus', { status: 'connected', userId });
    });

    socket.on('disconnect', () => {
      socketConnected = false;
      console.log('⚠️ Disconnected from real-time server');
      emitEvent('sessionStatus', { status: 'disconnected' });
    });

    socket.on('connect_error', (error) => {
      console.error('❌ Socket connection error:', error);
      emitEvent('error', { type: 'connection', error });
    });

    // Real-time event handlers
    socket.on('frame_update', (data) => {
      if (data) emitEvent('frameUpdate', sanitizeDataForSocket(data));
    });

    socket.on('metrics_update', (data) => {
      if (data) emitEvent('metricsUpdate', sanitizeDataForSocket(data));
    });

    socket.on('detection_alert', (data) => {
      if (data) emitEvent('detectionAlert', sanitizeDataForSocket(data));
    });

    socket.on('session_status', (data) => {
      if (data) {
        currentSessionId = data.session_id;
        emitEvent('sessionStatus', sanitizeDataForSocket(data));
      }
    });

    socket.on('error', (error) => {
      console.error('❌ Socket error:', error);
      emitEvent('error', { type: 'socket', error });
    });

    return true;
  } catch (error) {
    console.error('❌ Failed to initialize socket:', error);
    emitEvent('error', { type: 'init', error });
    return false;
  }
}

/**
 * Check if socket is connected
 */
export function isSocketConnected() {
  return socket && socketConnected;
}

/**
 * Reconnect socket if disconnected
 */
export function reconnectSocket() {
  if (!socket) {
    return initializeSocket();
  }
  
  if (socketConnected) {
    console.log('✅ Socket already connected');
    return true;
  }

  try {
    socket.connect();
    return true;
  } catch (error) {
    console.error('❌ Reconnection failed:', error);
    return false;
  }
}

/**
 * Disconnect socket
 */
export function disconnectSocket() {
  if (socket) {
    socket.disconnect();
    socketConnected = false;
    console.log('✅ Socket disconnected');
  }
}

/**
 * Start real-time detection
 */
export async function startRealtimeDetection(config) {
  if (!isSocketConnected()) {
    console.warn('⚠️ Socket not connected. Initializing...');
    if (!initializeSocket()) {
      throw new Error('Failed to connect socket');
    }
  }

  const userId = getCurrentUserId();
  if (!userId) {
    throw new Error('User not authenticated');
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Start detection timeout'));
    }, 10000);

    socket.emit(
      'start_realtime_detection',
      {
        user_id: userId,
        video_source: config.videoSource || '0',
        conf: config.confidence || 0.25,
        top_k: config.topK || 5,
        vlm_provider: config.vlmProvider || 'cosmos',
        vlm_max_crops: config.vlmMaxCrops || 5,
        vlm_min_yolo_conf: config.vlmMinYoloConf || 0.35,
        llm_provider: config.llmProvider || 'groq',
      },
      (response) => {
        clearTimeout(timeout);
        if (response && response.success) {
          currentSessionId = response.session_id;
          console.log('✅ Detection started. Session ID:', currentSessionId);
          resolve(response);
        } else {
          reject(new Error(response?.error || 'Failed to start detection'));
        }
      }
    );
  });
}

/**
 * Pause detection
 */
export async function pauseDetection() {
  if (!isSocketConnected()) {
    throw new Error('Socket not connected');
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Pause detection timeout'));
    }, 5000);

    socket.emit(
      'pause_detection',
      { user_id: getCurrentUserId(), session_id: currentSessionId },
      (response) => {
        clearTimeout(timeout);
        if (response && response.success) {
          console.log('✅ Detection paused');
          resolve(response);
        } else {
          reject(new Error(response?.error || 'Failed to pause detection'));
        }
      }
    );
  });
}

/**
 * Resume detection
 */
export async function resumeDetection() {
  if (!isSocketConnected()) {
    throw new Error('Socket not connected');
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Resume detection timeout'));
    }, 5000);

    socket.emit(
      'resume_detection',
      { user_id: getCurrentUserId(), session_id: currentSessionId },
      (response) => {
        clearTimeout(timeout);
        if (response && response.success) {
          console.log('✅ Detection resumed');
          resolve(response);
        } else {
          reject(new Error(response?.error || 'Failed to resume detection'));
        }
      }
    );
  });
}

/**
 * Stop detection and get final report
 */
export async function stopDetection() {
  if (!isSocketConnected()) {
    throw new Error('Socket not connected');
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Stop detection timeout'));
    }, 10000);

    socket.emit(
      'stop_detection',
      { user_id: getCurrentUserId(), session_id: currentSessionId },
      (response) => {
        clearTimeout(timeout);
        currentSessionId = null;
        if (response && response.success) {
          console.log('✅ Detection stopped');
          resolve(response);
        } else {
          reject(new Error(response?.error || 'Failed to stop detection'));
        }
      }
    );
  });
}

/**
 * Get session status via WebSocket
 */
export async function getSessionStatus() {
  if (!isSocketConnected()) {
    throw new Error('Socket not connected');
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Get session status timeout'));
    }, 5000);

    socket.emit(
      'get_session_status',
      { user_id: getCurrentUserId(), session_id: currentSessionId },
      (response) => {
        clearTimeout(timeout);
        if (response) {
          resolve(response);
        } else {
          reject(new Error('Failed to get session status'));
        }
      }
    );
  });
}

/**
 * Get detailed report via HTTP (after stopping)
 */
export async function getDetailedReport() {
  if (!currentSessionId) {
    throw new Error('No active session');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/sessions/${currentSessionId}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ Detailed report fetched');
    return data;
  } catch (error) {
    console.error('❌ Failed to get detailed report:', error);
    throw error;
  }
}

/**
 * Get RAG context for a detection
 */
export async function getRagContext(detection) {
  if (!currentSessionId) {
    throw new Error('No active session');
  }

  try {
    const response = await fetch(
      `${API_BASE_URL}/sessions/${currentSessionId}/rag-context`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sanitizeDataForSocket(detection)),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ RAG context retrieved');
    return data;
  } catch (error) {
    console.error('❌ Failed to get RAG context:', error);
    throw error;
  }
}

/**
 * Get RAG-enhanced threat analysis
 */
export async function getThreatAnalysis(detection) {
  if (!currentSessionId) {
    throw new Error('No active session');
  }

  try {
    const response = await fetch(
      `${API_BASE_URL}/sessions/${currentSessionId}/threat-analysis`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sanitizeDataForSocket(detection)),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('✅ Threat analysis retrieved');
    return data;
  } catch (error) {
    console.error('❌ Failed to get threat analysis:', error);
    throw error;
  }
}

/**
 * Get session crops (analyzed by VLM)
 */
export async function getSessionCrops() {
  if (!currentSessionId) {
    throw new Error('No active session');
  }

  try {
    const response = await fetch(`${API_BASE_URL}/sessions/${currentSessionId}/crops`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('❌ Failed to get crops:', error);
    throw error;
  }
}

/**
 * Get active sessions for current user
 */
export async function getActiveSessions() {
  const userId = getCurrentUserId();
  if (!userId) throw new Error('User not authenticated');

  try {
    const response = await fetch(
      `${API_BASE_URL}/sessions?user_id=${encodeURIComponent(userId)}`,
      {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('❌ Failed to get active sessions:', error);
    throw error;
  }
}

/**
 * Sanitize data for socket emission (prevents circular refs and large payloads)
 */
function sanitizeDataForSocket(obj, depth = 0, seen = new WeakSet()) {
  // Max depth to prevent stack overflow
  if (depth > 10) return undefined;

  // Handle null/undefined
  if (obj === null || obj === undefined) return obj;

  // Prevent circular references
  if (typeof obj === 'object') {
    if (seen.has(obj)) return '[Circular]';
    seen.add(obj);
  }

  // Handle primitives
  if (typeof obj !== 'object') return obj;

  // Skip binary data and large objects
  if (obj instanceof ArrayBuffer || obj instanceof Uint8Array) {
    return '[Binary Data]';
  }

  // Skip DOM nodes
  if (obj instanceof HTMLElement || obj instanceof Node) {
    return '[DOM Node]';
  }

  // Handle arrays
  if (Array.isArray(obj)) {
    // Limit array size to prevent huge payloads
    return obj.slice(0, 100).map((item) => sanitizeDataForSocket(item, depth + 1, seen));
  }

  // Handle objects
  if (obj instanceof Object) {
    const sanitized = {};
    let count = 0;

    for (const key in obj) {
      if (obj.hasOwnProperty(key) && count < 50) {
        // Limit object size
        // Skip problematic keys
        if (key === '_crop_image' || key === 'crop_image' || key === 'image') {
          sanitized[key] = '[Image Data]';
        } else if (key === 'file' || key === 'buffer') {
          sanitized[key] = '[File Data]';
        } else {
          sanitized[key] = sanitizeDataForSocket(obj[key], depth + 1, seen);
        }
        count++;
      }
    }

    return sanitized;
  }

  return obj;
}

/**
 * Emit events to listeners
 */
function emitEvent(eventType, data) {
  if (eventListeners[eventType]) {
    eventListeners[eventType].forEach((callback) => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Error in ${eventType} listener:`, error);
      }
    });
  }
}

/**
 * Register event listeners
 */
export function onFrameUpdate(callback) {
  eventListeners.frameUpdate.push(callback);
  return () => {
    eventListeners.frameUpdate = eventListeners.frameUpdate.filter((cb) => cb !== callback);
  };
}

export function onMetricsUpdate(callback) {
  eventListeners.metricsUpdate.push(callback);
  return () => {
    eventListeners.metricsUpdate = eventListeners.metricsUpdate.filter((cb) => cb !== callback);
  };
}

export function onDetectionAlert(callback) {
  eventListeners.detectionAlert.push(callback);
  return () => {
    eventListeners.detectionAlert = eventListeners.detectionAlert.filter((cb) => cb !== callback);
  };
}

export function onSessionStatus(callback) {
  eventListeners.sessionStatus.push(callback);
  return () => {
    eventListeners.sessionStatus = eventListeners.sessionStatus.filter((cb) => cb !== callback);
  };
}

export function onSocketError(callback) {
  eventListeners.error.push(callback);
  return () => {
    eventListeners.error = eventListeners.error.filter((cb) => cb !== callback);
  };
}

/**
 * Unregister all listeners
 */
export function offFrameUpdate(callback) {
  eventListeners.frameUpdate = eventListeners.frameUpdate.filter((cb) => cb !== callback);
}

export function offMetricsUpdate(callback) {
  eventListeners.metricsUpdate = eventListeners.metricsUpdate.filter((cb) => cb !== callback);
}

export function offDetectionAlert(callback) {
  eventListeners.detectionAlert = eventListeners.detectionAlert.filter((cb) => cb !== callback);
}

/**
 * Data transformation utilities
 */

export function formatDetection(detection) {
  if (!detection) return null;

  return {
    id: detection.frame || detection.track_id,
    time: detection.time_sec || 0,
    label: detection.label || 'unknown',
    confidence: detection.confidence || detection.yolo_confidence || 0,
    trackId: detection.track_id,
    threatLevel: detection.vlm_analysis?.threat_level || 'none',
    threatScore: detection.unified_score || 0,
    payload: detection.vlm_analysis?.payload_detected || false,
  };
}

export function metricsToChartData(metrics) {
  if (!metrics || !metrics.timeline_detections_per_second) return [];

  const timeline = metrics.timeline_detections_per_second;
  return Array.isArray(timeline)
    ? timeline
    : Object.entries(timeline).map(([time, count]) => ({
        time: Number(time),
        detections: count,
      }));
}

export function perClassTimelineToChartData(metrics) {
  if (!metrics?.timeline_detections_per_second_per_class) return [];

  const timelinePerClass = metrics.timeline_detections_per_second_per_class;
  const allTimestamps = new Set();

  // Collect all unique timestamps
  Object.values(timelinePerClass).forEach((timeline) => {
    Object.keys(timeline).forEach((time) => {
      allTimestamps.add(Number(time));
    });
  });

  // Create chart data
  return Array.from(allTimestamps)
    .sort((a, b) => a - b)
    .map((time) => {
      const dataPoint = { time };
      Object.entries(timelinePerClass).forEach(([className, timeline]) => {
        dataPoint[className] = timeline[time] || 0;
      });
      return dataPoint;
    });
}

export function classCountsToChartData(metrics) {
  if (!metrics?.class_counts) return [];

  return Object.entries(metrics.class_counts).map(([name, count]) => ({
    name,
    count,
  }));
}

export function threatCountsToChartData(metrics) {
  if (!metrics?.vlm_analysis?.threat_level_counts) return [];

  return Object.entries(metrics.vlm_analysis.threat_level_counts).map(([name, count]) => ({
    name,
    count,
  }));
}

export function tracksToChartData(metrics) {
  if (!metrics?.tracking?.top_tracks) return [];

  return metrics.tracking.top_tracks.map((track) => ({
    track_id: `${track.label} (${track.track_id})`,
    duration_sec: track.duration_sec,
    appearances: track.detection_count,
  }));
}

/**
 * Calculate statistics
 */
export function calculateStatistics(metrics) {
  if (!metrics) return {};

  const totalDetections = metrics.total_detections || 0;
  const framesWithDetections = metrics.frames_with_detections || 0;
  const totalFrames = metrics.total_frames || 1;

  return {
    detectionRate: totalFrames ? (framesWithDetections / totalFrames) * 100 : 0,
    avgDetectionsPerFrame: totalFrames ? totalDetections / totalFrames : 0,
    maxDetectionsInFrame: metrics.max_detections_in_frame || 0,
    uniqueTracks: metrics.tracking?.unique_tracks || 0,
    avgTrackDuration: metrics.tracking?.avg_track_duration_sec || 0,
  };
}

export function calculateFalsePositiveRate(metrics) {
  const total = metrics?.vlm_analysis?.analyzed_crops || 0;
  const none = metrics?.vlm_analysis?.threat_level_counts?.none || 0;

  if (total === 0) return 0;
  return ((none / total) * 100).toFixed(1);
}

export function getTopDetections(metrics, limit = 5) {
  return (metrics?.top_confidence_events || []).slice(0, limit);
}

/**
 * Export utilities
 */
export function downloadSessionReport(report, sessionId) {
  const json = JSON.stringify(report, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `detection_report_${sessionId}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadSessionCSV(metrics, sessionId) {
  if (!metrics?.top_confidence_events) {
    console.warn('No events to export');
    return;
  }

  const headers = ['Time', 'Label', 'Confidence', 'Threat Level', 'Track ID'];
  const rows = metrics.top_confidence_events.map((event) => [
    event.time_sec,
    event.label,
    event.confidence,
    event.vlm_analysis?.threat_level || 'N/A',
    event.track_id || 'N/A',
  ]);

  const csv = [
    headers.join(','),
    ...rows.map((row) => row.map((cell) => `"${cell}"`).join(',')),
  ].join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `detections_${sessionId}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Formatting utilities
 */
export function formatThreatLevel(level) {
  const levels = {
    none: { color: '#10b981', label: 'None' },
    low: { color: '#f59e0b', label: 'Low' },
    medium: { color: '#f97316', label: 'Medium' },
    high: { color: '#ef4444', label: 'High' },
    critical: { color: '#991b1b', label: 'Critical' },
  };

  return levels[level?.toLowerCase()] || levels.none;
}

export function getClassColor(className) {
  const colors = {
    drone: '#1f10f1',
    bird: '#ff6b6b',
    airplane: '#4ecdc4',
    helicopter: '#ffa500',
    other: '#95e1d3',
  };

  return colors[className?.toLowerCase()] || '#8b5cf6';
}

export default {
  initializeSocket,
  isSocketConnected,
  reconnectSocket,
  disconnectSocket,
  getCurrentUserId,
  startRealtimeDetection,
  pauseDetection,
  resumeDetection,
  stopDetection,
  getSessionStatus,
  getDetailedReport,
  getRagContext,
  getThreatAnalysis,
  getSessionCrops,
  getActiveSessions,
  onFrameUpdate,
  onMetricsUpdate,
  onDetectionAlert,
  onSessionStatus,
  onSocketError,
  offFrameUpdate,
  offMetricsUpdate,
  offDetectionAlert,
  formatDetection,
  metricsToChartData,
  perClassTimelineToChartData,
  classCountsToChartData,
  threatCountsToChartData,
  tracksToChartData,
  calculateStatistics,
  calculateFalsePositiveRate,
  getTopDetections,
  downloadSessionReport,
  downloadSessionCSV,
  formatThreatLevel,
  getClassColor,
};
