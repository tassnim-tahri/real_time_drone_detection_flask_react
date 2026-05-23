import express from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import Report from "../models/Report.js";

const router = express.Router();

const FLASK_AI_URL = (process.env.FLASK_AI_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
  },
});

const upload = multer({ storage });

function mapDetectionToReport(d) {
  const bb = d.bbox || {};
  return {
    detected_object: d.label || "unknown",
    axe_x: typeof bb.x === "number" ? bb.x : 0,
    axe_y: typeof bb.y === "number" ? bb.y : 0,
    width: typeof bb.width === "number" ? bb.width : 0,
    height: typeof bb.height === "number" ? bb.height : 0,
  };
}

router.post("/detect", upload.any(), async (req, res) => {
  const file = req.files?.[0];
  let tmpPath;

  try {
    if (!file) {
      return res.status(400).json({
        message: "No file uploaded (use multipart field \"file\" or \"video\")",
      });
    }

    tmpPath = file.path;
    console.log("Uploaded file:", file.originalname);

    const formData = new FormData();
    formData.append("file", fs.createReadStream(tmpPath), file.originalname);

    const flaskResponse = await axios.post(`${FLASK_AI_URL}/detect`, formData, {
      headers: formData.getHeaders(),
    });

    console.log("Flask response:", flaskResponse.data);

    const body = flaskResponse.data;
    const list = Array.isArray(body.detections)
      ? body.detections
      : Array.isArray(body)
        ? body
        : [];

    const reports = list.map(mapDetectionToReport);
    if (reports.length > 0) {
      await Report.insertMany(reports);
    }

    res.json({
      message: "Detection completed successfully",
      flask: body,
      detections_saved: reports.length,
    });
  } catch (err) {
    console.error("FULL ERROR:", err?.response?.data || err.message);
    res.status(500).json({
      message: err.message,
      flask_error: err?.response?.data,
    });
  } finally {
    if (tmpPath && fs.existsSync(tmpPath)) {
      try {
        fs.unlinkSync(tmpPath);
      } catch (_) {
        /* ignore */
      }
    }
  }
});

export default router;
