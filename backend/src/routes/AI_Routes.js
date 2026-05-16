import express from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import Report from "../models/Report.js";

const router = express.Router();

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname); // keep original name + extension
  }
});

const upload = multer({ storage });

router.post("/detect", upload.single("video"), async (req, res) => {
  try {
    console.log("Received request");

    if (!req.file) {
      return res.status(400).json({
        message: "No file uploaded"
      });
    }

    console.log("Uploaded file:", req.file);

    const formData = new FormData();

    formData.append(
      "video",
      fs.createReadStream(req.file.path),
      req.file.originalname
    );

    const flaskResponse = await axios.post(
      "http://192.168.1.17:8000/detect",
      formData,
      {
        headers: formData.getHeaders()
      }
    );

    console.log("Flask response:", flaskResponse.data);

    const detections = flaskResponse.data;

    if (detections.length > 0) {
      await Report.insertMany(detections);
    }

    res.json({
      message: "Detection completed successfully",
      detections
    });

  } catch (err) {
    console.log("FULL ERROR:");
    console.log(err);

    res.status(500).json({
      message: err.message
    });
  }
});

export default router;
