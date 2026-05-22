import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  initializeSocket,
  isSocketConnected,
  requestCameraAccess,
  stopCameraStream,
  captureAndSendFrame,
  startRealtimeDetection,
  pauseDetection,
  resumeDetection,
  stopDetection,
  getDetailedReport,
  onMetricsUpdate,
  offMetricsUpdate,
  getCurrentSessionId,
  metricsToChartData,
  classCountsToChartData,
  getTopDetections,
} from '../../services/RT_ai_services';

import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

const COLORS = ['#3b82f6', '#1d4ed8', '#60a5fa', '#2563eb', '#ef4444', '#10b981'];

export default function RTDashboard() {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  // State management
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [metrics, setMetrics] = useState({
    frame_count: 0,
    detection_count: 0,
    total_detections: 0,
    class_counts: {},
    top_confidence_events: [],
  });
  const [error, setError] = useState(null);
  const [confidence, setConfidence] = useState(0.25);
  const [socketReady, setSocketReady] = useState(false);

  // Initialize socket on mount
  useEffect(() => {
    const connected = initializeSocket();
    setSocketReady(connected);
    
    if (!connected) {
      setError('Failed to connect to server');
    }

    return () => {
      stopCameraStream();
    };
  }, []);

  // Listen for metrics updates
  useEffect(() => {
    const unsubscribe = onMetricsUpdate((newMetrics) => {
      setMetrics(newMetrics);
    });

    return () => unsubscribe();
  }, []);

  // Draw video to canvas and send frames
  const startFrameCapture = async () => {
    try {
      // Request camera access
      const stream = await requestCameraAccess(canvasRef.current);
      
      // Set up video element
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          captureFrames();
        };
      }
    } catch (err) {
      console.error('Camera error:', err);
      setError(`Camera access denied: ${err.message}`);
    }
  };

  // Continuously capture and send frames
  const captureFrames = () => {
    if (!isRunning || !canvasRef.current || !videoRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Send frame to backend at ~10 FPS to avoid overwhelming server
    if (Math.random() < 0.33) { // ~1 in 3 frames
      captureAndSendFrame();
    }

    animationRef.current = requestAnimationFrame(captureFrames);
  };

  // Start detection
  const handleStart = async () => {
    try {
      setError(null);

      if (!socketReady) {
        setError('Socket not connected');
        return;
      }

      // Start backend detection
      await startRealtimeDetection({
        videoSource: 'client_camera', // ✅ Client-side camera
        confidence: confidence,
        vlmProvider: 'cosmos',
        llmProvider: 'groq',
      });

      setIsRunning(true);
      setIsPaused(false);

      // Start capturing frames from local camera
      await startFrameCapture();
    } catch (err) {
      console.error('Start error:', err);
      setError(err.message);
      setIsRunning(false);
    }
  };

  // Pause detection
  const handlePause = async () => {
    try {
      await pauseDetection();
      setIsPaused(true);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  // Resume detection
  const handleResume = async () => {
    try {
      await resumeDetection();
      setIsPaused(false);
      captureFrames();
    } catch (err) {
      setError(err.message);
    }
  };

  // Stop detection
  const handleStop = async () => {
    try {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      stopCameraStream();

      await stopDetection();
      setIsRunning(false);
      setIsPaused(false);

      // Get final report and navigate
      const sessionId = getCurrentSessionId();
      if (sessionId) {
        const report = await getDetailedReport(sessionId);
        navigate('/dashboard', { state: { report } });
      }
    } catch (err) {
      console.error('Stop error:', err);
      setError(err.message);
    }
  };

  // Chart data transformations
  const timelineData = metricsToChartData(metrics);
  const classData = classCountsToChartData(metrics);
  const topEvents = getTopDetections(metrics, 5);

  const cardClass = "bg-gray-800 p-4 rounded-lg border border-blue-500 text-center";
  const cardValue = "text-2xl font-bold text-blue-400";

  return (
    <div className="bg-gray-950 text-white min-h-screen p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">🎥 Real-Time Drone Detection</h1>
        <p className="text-gray-400">
          {isRunning ? '🔴 LIVE' : '⚫ STOPPED'} | Camera: {isRunning ? 'Active' : 'Inactive'}
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-4 bg-red-900 border border-red-500 rounded">
          <p className="text-red-200">❌ {error}</p>
        </div>
      )}

      {/* Video Stream & Controls */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        {/* Camera Feed */}
        <div className="col-span-2">
          <div className="bg-gray-900 rounded-lg overflow-hidden border-2 border-blue-500">
            <div className="relative">
              {/* Hidden video element for camera input */}
              <video
                ref={videoRef}
                style={{ display: 'none' }}
                autoPlay
                playsInline
              />
              
              {/* Canvas for displaying frames */}
              <canvas
                ref={canvasRef}
                className="w-full aspect-video bg-black"
              />
              
              {/* Status overlay */}
              <div className="absolute top-4 right-4 bg-black bg-opacity-70 px-4 py-2 rounded">
                {isRunning && <span className="text-green-400">🔴 LIVE</span>}
                {isPaused && <span className="text-yellow-400">⏸️ PAUSED</span>}
                {!isRunning && <span className="text-gray-400">⚫ INACTIVE</span>}
              </div>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-gray-900 p-6 rounded-lg border border-blue-500">
          <h2 className="text-xl font-bold mb-4">Controls</h2>

          {/* Confidence Slider */}
          <div className="mb-6">
            <label className="block text-sm mb-2">Confidence: {(confidence * 100).toFixed(0)}%</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              disabled={isRunning}
              className="w-full"
            />
          </div>

          {/* Buttons */}
          <div className="space-y-2">
            {!isRunning ? (
              <button
                onClick={handleStart}
                className="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded font-bold"
              >
                ▶️ START
              </button>
            ) : (
              <>
                {!isPaused ? (
                  <button
                    onClick={handlePause}
                    className="w-full bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded font-bold"
                  >
                    ⏸️ PAUSE
                  </button>
                ) : (
                  <button
                    onClick={handleResume}
                    className="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-bold"
                  >
                    ▶️ RESUME
                  </button>
                )}
                <button
                  onClick={handleStop}
                  className="w-full bg-red-600 hover:bg-red-700 px-4 py-2 rounded font-bold"
                >
                  ⏹️ STOP
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <div className={cardClass}>
          <p className="text-gray-400 mb-2">Frames Processed</p>
          <p className={cardValue}>{metrics.frame_count || 0}</p>
        </div>
        <div className={cardClass}>
          <p className="text-gray-400 mb-2">Detections (Current)</p>
          <p className={cardValue}>{metrics.detection_count || 0}</p>
        </div>
        <div className={cardClass}>
          <p className="text-gray-400 mb-2">Total Detections</p>
          <p className={cardValue}>{metrics.total_detections || 0}</p>
        </div>
        <div className={cardClass}>
          <p className="text-gray-400 mb-2">Unique Classes</p>
          <p className={cardValue}>{Object.keys(metrics.class_counts || {}).length}</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6 mb-8">
        {/* Timeline Chart */}
        <div className="bg-gray-900 p-4 rounded-lg border border-blue-500">
          <h3 className="text-lg font-bold mb-4">📈 Detections Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="detections" fill="#3b82f6" stroke="#1f10f1" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Class Distribution */}
        <div className="bg-gray-900 p-4 rounded-lg border border-blue-500">
          <h3 className="text-lg font-bold mb-4">🏷️ Class Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={classData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Top Events Table */}
      <div className="bg-gray-900 p-4 rounded-lg border border-blue-500">
        <h3 className="text-lg font-bold mb-4">⭐ Top Detections</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-blue-500">
              <tr>
                <th className="text-left py-2 px-4">Time</th>
                <th className="text-left py-2 px-4">Label</th>
                <th className="text-left py-2 px-4">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {topEvents.map((event, idx) => (
                <tr key={idx} className="border-b border-gray-700 hover:bg-gray-800">
                  <td className="py-2 px-4">{event.time_sec?.toFixed(2)}s</td>
                  <td className="py-2 px-4">{event.label}</td>
                  <td className="py-2 px-4">{(event.confidence * 100).toFixed(1)}%</td>
                </tr>
              ))}
              {topEvents.length === 0 && (
                <tr>
                  <td colSpan="3" className="text-center py-4 text-gray-500">
                    No detections yet...
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
