import express from "express";
import {
  getAllReports
} from "../controllers/reportController.js";
import e from "express";

const report_router = express.Router();

report_router.get("/", getAllReports);

export default report_router;