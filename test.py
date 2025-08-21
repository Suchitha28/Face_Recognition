import face_recognition
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import cvzone
import torch
from facenet_pytorch import MTCNN
import threading
import queue
import db 
from attendance_notification import send_sms_to_employee, schedule_daily_email,send_admin_alert_sms
import mysql.connector
from db import get_local_connection
import threading
import time
import warnings
import logging
import subprocess
import atexit
import requests
import time
import sys
import os
import cv2

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress ultralytics (YOLO) logging
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_VERBOSITY"] = "ERROR"

# Suppress PyTorch logging
logging.getLogger('torch').setLevel(logging.ERROR)

# Suppress general logging
logging.basicConfig(level=logging.ERROR)

# ========== CONFIG ==========
RTSP_STREAM_URL = 'rtsp://user123:password123@192.168.0.104:554/stream1?tcp'
frame_skip = 5
display_size = (640, 360)
frame_queue = queue.Queue(maxsize=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

face_detector = MTCNN(keep_all=True, device=device)
anti_spoofing_model_1 = YOLO('models/yolov8n.pt')
# anti_spoofing_model_2 = YOLO('models/n_version_1_30.pt')
anti_spoofing_model_2 = YOLO('models/yolov8n.pt')


known_face_encodings = []
known_face_names = []

# Path to the FastAPI script
FASTAPI_URL = "http://127.0.0.1:8000/replicate/"
FASTAPI_DOCS_URL = "http://127.0.0.1:8000/docs"

# Start FastAPI server
try:
    fastapi_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cloud_api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
except Exception as e:
    print("Error launching FastAPI server:", e)
    fastapi_process = None

# Ensure it's killed when this script exits
def cleanup():
    if fastapi_process:
        fastapi_process.terminate()
        print("FastAPI server terminated.")

atexit.register(cleanup)

# Wait until the FastAPI server is up
def wait_for_fastapi(url=FASTAPI_DOCS_URL, timeout=30):
    print("Waiting for FastAPI to start...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("FastAPI server is up!")
                return True
        except:
            pass
        time.sleep(1)
    print("Failed to connect to FastAPI server. Proceeding anyway.")
    return False

wait_for_fastapi()
#------------fast api ends------------


def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def detect_faces_mtcnn(frame):
    resized_frame = cv2.resize(frame, (720, 405), interpolation=cv2.INTER_LINEAR)
    boxes, probs = face_detector.detect(resized_frame)
    face_boxes = []
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob >= 0.85:
                x1, y1, x2, y2 = map(int, box)
                scale_x = frame.shape[1] / 720
                scale_y = frame.shape[0] / 405
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                face_boxes.append((x1, y1, x2, y2))
    return face_boxes

def detect_spoofing_ensemble(frame, face_boxes, confidence=0.85, min_face_area=5000):
    spoof_detected = False

    for (x1, y1, x2, y2) in face_boxes:
        w, h = x2 - x1, y2 - y1
        if w * h < min_face_area:
            continue

        face_crop = frame[y1:y2, x1:x2]

        # Resize face crop for YOLO models to expected input size
        face_crop_resized = cv2.resize(face_crop, (640, 640))  # Assuming YOLO expects 640x640 input

        # Perform anti-spoofing detection with YOLO models
        try:
            results1 = anti_spoofing_model_1([face_crop_resized], verbose=False)  # Run YOLO model 1
            results2 = anti_spoofing_model_2([face_crop_resized], verbose=False)  # Run YOLO model 2
        except Exception as e:
            print(f"Error during spoofing detection: {e}")
            continue  # Skip current face if any error occurs

        labels = []
        for results in [results1, results2]:
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])  # Assuming class 0 is 'real' and class 1 is 'fake'
                    if conf >= confidence:
                        labels.append(cls)

        # Consensus for spoofing detection
        if labels.count(1) >= 2:  # Majority vote for fake detection
            cvzone.putTextRect(frame, 'FAKE (Consensus)', (x1, y1 - 10), scale=1, thickness=2, colorR=(0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            spoof_detected = True
        else:
            cvzone.putTextRect(frame, 'REAL (Consensus)', (x1, y1 - 10), scale=1, thickness=2, colorR=(0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return 'FAKE' if spoof_detected else 'REAL'

def load_known_faces():
    folder_path = 'retrieved_employees/'
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.png')):
            name_id = os.path.splitext(file_name)[0]
            try:
                name, emp_id = name_id.rsplit('_', 1)  # Expects format: Name_EmpID.jpg
                name = name.replace('_', ' ')  # Convert "Jane_Doe" → "Jane Doe"
                img_path = os.path.join(folder_path, file_name)

                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append((name, emp_id))
                else:
                    print(f"No face found in {img_path}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

def mark_attendance_to_mysql(name, emp_id):
    connection = get_local_connection()
    if connection:
        try:
            cursor = connection.cursor()

            # ✅ Check if the employee is active
            cursor.execute("SELECT active_status FROM status WHERE emp_id = %s", (emp_id,))
            status_row = cursor.fetchone()

            if not status_row or status_row[0].lower() != 'active':
                print(f"Cannot mark attendance: Employee {emp_id} is not active.")
                return  # ❗ Exit early if employee is not active

            today = datetime.now().date()

            # ❓ Check if attendance already marked today
            cursor.execute(
                "SELECT * FROM attendance WHERE emp_id = %s AND DATE(timestamp) = %s",
                (emp_id, today)
            )
            result = cursor.fetchone()

            if result:
                print(f"Attendance for employee {emp_id} already recorded today.")
                return  # ❗ Exit early, do NOT send SMS again

            # ✅ Insert attendance record
            timestamp = datetime.now()
            cursor.execute(
                "INSERT INTO attendance (emp_id, name, timestamp) VALUES (%s, %s, %s)",
                (emp_id, name, timestamp)
            )
            connection.commit()
            print("Record inserted into local database successfully.")

            # 📲 Send SMS only if inserted
            send_sms_to_employee(emp_id, name)

        except mysql.connector.Error as err:
            print(f"MySQL Error: {err}")

        finally:
            cursor.close()
            connection.close()

def frame_capture_thread():
    cap = cv2.VideoCapture(RTSP_STREAM_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print("Failed to connect to the RTSP stream! Exiting.")
        cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret and not frame_queue.full():
            frame_queue.put(frame)

def send_to_cloud(emp_id, name, timestamp):
    payload = {
        "emp_id": emp_id,
        "name": name,
        "timestamp": timestamp
    }
    try:
        response = requests.post(FASTAPI_URL, json=payload)
        if response.status_code == 200:
            print("Record replicated to S3 successfully!")
        else:
            print("Failed to replicate record to S3.")
    except Exception as e:
        print(f"Error sending data to S3: {e}")

def run_attendance_system():
    load_known_faces()
    print("Starting attendance system...")
    threading.Thread(target=frame_capture_thread, daemon=True).start()

    frame_count = 0
    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = enhance_contrast(frame)
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)

        face_boxes = detect_faces_mtcnn(frame)

        if face_boxes and frame_count % (frame_skip * 3) == 0:
            spoof_result = detect_spoofing_ensemble(frame, face_boxes)
            if spoof_result == 'FAKE':
                print("Access denied due to spoofing.")
                continue

        for (x1, y1, x2, y2) in face_boxes:
            face_crop = frame[y1:y2, x1:x2]

            if face_crop is None or face_crop.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(face_rgb)

            name_display = "Unknown"
            emp_id = None
            name = None

            if encodings:
                face_encoding = encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if matches and True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name, emp_id = known_face_names[best_match_index]
                        name_display = f"{name} ({emp_id})"

                        # ✅ Mark attendance locally
                        mark_attendance_to_mysql(name, emp_id)

                        # ☁️ Send to cloud
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        send_to_cloud(emp_id, name, timestamp)

                        # 🔔 Send alert to admin
                        send_admin_alert_sms(emp_id, name)
                    else:
                        name_display = "Unknown"

            # 🖼️ Draw result
            color = (0, 255, 0) if name_display != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name_display, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        resized_display = cv2.resize(frame, display_size)
        cv2.imshow('Attendance System', resized_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Real-Time Face Recognition Attendance System with Notifications...")
    # Start email scheduler (default 6:00 PM)
    schedule_daily_email(hour=10,minute=43)
    # Start attendance system loop
    run_attendance_system()
