from unittest import result
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from app import run_full_pipeline
import uuid
import traceback
import json

# =============================
# INIT APP
# =============================

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model once
model = YOLO("model.pt")

chain_list={}
reportText_list={}
chain = None
reportText=None

def save_file(file):
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "AI running"})

import subprocess

OUTPUT_REPORT_PATH = "outputs/final_threat_report.md"

import re

def parse_report(report_text):
    data = {}

    # ------------------ QUICK SUMMARY ------------------
    qs = {}
    patterns = {
        "drones_detected": r"Drones detected:\s*(\d+)",
        "time_monitored": r"Time monitored:\s*([\d.]+)",
        "frames_with_drones": r"Frames containing drones:\s*(\d+)",
        "unique_drones": r"Unique drones tracked:\s*(\d+)",
        "risk_level": r"Overall risk level:\s*(\w+)",
        "highest_risk": r"Highest risk category observed:\s*(\w+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, report_text)
        if match:
            qs[key] = match.group(1)

    data["quick_summary"] = qs

    # ------------------ MAIN FINDINGS ------------------
    mf = {}
    patterns2 = {
        "drone_type": r"Most common drone type:\s*(.+)",
        "payload": r"Most common payload type:\s*(.+)",
        "size": r"Common drone size:\s*(.+)",
        "threat": r"Threat distribution:\s*(.+)",
    }

    for key, pattern in patterns2.items():
        match = re.search(pattern, report_text)
        if match:
            mf[key] = match.group(1)

    data["main_findings"] = mf

    # ------------------ ACTIONS ------------------
    actions = re.findall(r"- (.+)", report_text)
    data["actions"] = actions

    # ------------------ SUMMARY TEXT ------------------
    summary_match = re.search(r"Plain-Language Summary\n\n(.+)", report_text, re.DOTALL)
    if summary_match:
        data["summary"] = summary_match.group(1).strip()

    return data


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    print(request.form)
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    if "conf" not in request.form:
        return jsonify({"success": False, "error": "No conf sent"}), 400
    if "top_K" not in request.form:
        return jsonify({"success": False, "error": "No top_K sent"}), 400
    if "llmProvider" not in request.form:
        return jsonify({"success": False, "error": "No LLM provider sent"}), 400
    if "user_id" not in request.form:
        return jsonify({"success": False, "error": "No User ID sent"}), 400
    
    video = request.files["file"]
    conf = float(request.form["conf"])
    top_K = int(request.form["top_K"])
    llmProvider = str(request.form["llmProvider"])
    user_id= str(request.form["user_id"])
    print(user_id)
    filename = secure_filename(video.filename)
    input_path = os.path.join("uploads", filename)
    video.save(input_path)

    try:
        
        global chain_list
        global reportText_list
        result,chain = run_full_pipeline(input_path,conf,top_K,"groq","cosmos",os.getenv("HF_KEY"))
        chain_list[user_id] = chain
        reportText = result["raw_report"]
        reportText_list[user_id] = reportText
        return jsonify({
            "success": True,
            "report_path": result["report_path"],
            "report_text": result["report_text"],
            "report_from": result["report_from"],
            "report": result["raw_report"],
            "summary": result["summary"]
        })
    except Exception as e:
        print("Pipeline error:", str(e))
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Pipeline failed: {str(e)}"}), 500

@app.route("/getlists" , methods=["GET"])
def getlists():
     return jsonify({
        "report": reportText_list,
        "chain_types": [
            str(type(c)) for c in chain_list
        ]
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        
        data = request.json
        user_message = data.get("message", "")
        user_id = data.get("user_id","")
        if not user_message or not user_id : 
            return jsonify({"error": "no user_id or session not initialized"}), 400
        global chain_list , reportText_list
        chain = chain_list[user_id]
        reportText = reportText_list[user_id]
        if chain is None or reportText is None:
            return jsonify({"error": "Invalid user_id or session not initialized"}), 400
        print("chain : ",chain)
        try : 
            response = chain.invoke({
                "report": reportText,"question": user_message
            })
            print("chain : ",chain)
            return jsonify({
                "response": response
            })
        except Exception as exc:
            print(f"Assistant: Could not answer with LangChain provider  yayyy: {exc}\n")
            return jsonify({"response" : f"Assistant: Could not answer with LangChain provider: {exc}\n "})

    except Exception as e:
        print("ERROR",str(e))
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
