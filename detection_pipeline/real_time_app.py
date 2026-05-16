import cv2
import time
import threading
from ultralytics import YOLO
# LOAD MODEL

MODEL_PATH = "weights/model.pt"

model = YOLO(MODEL_PATH)
# GLOBAL LIVE STATE

LIVE_STATS = {
    "running": False,
    "fps": 0,
    "detections": 0,
    "current_frame": 0,
    "class_counts": {},
    "last_detection": None
}

LATEST_FRAME = None

# DETECTION LOOP

def start_realtime_detection(camera_index=0):

    global LIVE_STATS
    global LATEST_FRAME
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    LIVE_STATS["running"] = True
    prev_time = time.time()
    frame_count = 0
    print("Realtime detection started")
    while LIVE_STATS["running"]:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        # YOLO INFERENCE

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )
        annotated_frame = results[0].plot()
        detections_this_frame = 0
        

        # PARSE DETECTIONS

        if results[0].boxes is not None:
            boxes = results[0].boxes
            detections_this_frame = len(boxes)
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                LIVE_STATS["class_counts"][class_name] = (
                    LIVE_STATS["class_counts"].get(class_name, 0) + 1
                )
                LIVE_STATS["last_detection"] = class_name

        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        # ====================================================
        # UPDATE LIVE STATS
        # ====================================================
        LIVE_STATS["fps"] = round(fps, 2)
        LIVE_STATS["detections"] += detections_this_frame
        LIVE_STATS["current_frame"] = frame_count
        LATEST_FRAME = annotated_frame
        # ====================================================
        # OPTIONAL LOCAL DISPLAY
        # ====================================================
        #cv2.imshow("Realtime Detection", annotated_frame)

        #if cv2.waitKey(1) & 0xFF == ord("q"):
         #   break

    LIVE_STATS["running"] = False

    cap.release()
    #cv2.destroyAllWindows()

    print("Realtime detection stopped")

# START THREAD

def launch_detection_thread():
    thread = threading.Thread(
        target=start_realtime_detection,
        daemon=True
    )

    thread.start()

    return thread

