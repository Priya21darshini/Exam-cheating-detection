import streamlit as st
import cv2
import numpy as np
import time
from detectors import Detector
from tracker import Tracker
from head_pose import HeadPoseEstimator
from recorder import Recorder
from collections import deque

st.title("AI Exam Cheating Detection (Live Webcam)")

# Initialize modules
detector = Detector()
tracker = Tracker()
pose_estimator = HeadPoseEstimator()
recorder = Recorder()

# Auto-calibration
calibration_time_sec = 5
start_time = time.time()
yaw_values = []
pitch_values = []
auto_calibrated = False
YAW_THRESHOLD, PITCH_THRESHOLD = 20, 15  # initial guess

# Frame smoothing
SMOOTHING_FRAMES = 5
yaw_history = deque(maxlen=SMOOTHING_FRAMES)
pitch_history = deque(maxlen=SMOOTHING_FRAMES)

# Labels and frame counter
labels = {}
frame_id = 0

# Streamlit placeholder
video_placeholder = st.empty()

# Open webcam
cap = cv2.VideoCapture(0)
st.text("Press 'q' to quit the detection.")

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Cannot access webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- 1) Detect person and phone ---
    detections = detector.detect(frame)
    person_boxes = [d[0] for d in detections if d[1] == 0]
    tracked = tracker.update(person_boxes)

    # --- 2) Head pose estimation ---
    poses = pose_estimator.estimate(frame)

    # --- 3) Match poses to tracked IDs ---
    sid2pose = {}
    for sid, pbox in tracked:
        best_iou, best_pose = 0, None
        for pose in poses:
            hbox = pose["bbox"]
            x1 = max(pbox[0], hbox[0]); y1 = max(pbox[1], hbox[1])
            x2 = min(pbox[2], hbox[2]); y2 = min(pbox[3], hbox[3])
            inter = max(0, x2-x1)*max(0, y2-y1)
            areaP = (pbox[2]-pbox[0])*(pbox[3]-pbox[1])+1e-6
            areaH = (hbox[2]-hbox[0])*(hbox[3]-hbox[1])+1e-6
            iou = inter/(areaP+areaH-inter)
            if iou > best_iou:
                best_iou, best_pose = iou, pose
        if best_pose and best_iou > 0.1:
            sid2pose[sid] = best_pose

    # --- Auto-calibration for first few seconds ---
    elapsed = time.time() - start_time
    if not auto_calibrated and elapsed <= calibration_time_sec:
        for pose in sid2pose.values():
            yaw_values.append(pose["yaw"])
            pitch_values.append(pose["pitch"])
        cv2.putText(frame, "CALIBRATING... Sit normally", (50,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    elif not auto_calibrated:
        max_yaw = max(abs(v) for v in yaw_values) if yaw_values else 0
        max_pitch = max(abs(v) for v in pitch_values) if pitch_values else 0
        YAW_THRESHOLD = max(max_yaw+10, 20)
        PITCH_THRESHOLD = max(max_pitch+10, 15)
        auto_calibrated = True
        st.success(f"Calibration done! yaw_th={YAW_THRESHOLD:.1f}, pitch_th={PITCH_THRESHOLD:.1f}")

    # --- Phone detection ---
    phone_sid = {}
    for box, cls in detections:
        if cls == 67:
            cx = (box[0]+box[2])/2
            cy = (box[1]+box[3])/2
            for sid, pbox in tracked:
                if pbox[0]<=cx<=pbox[2] and pbox[1]<=cy<=pbox[3]:
                    phone_sid[sid] = True

    # --- Cheating detection with smoothing ---
    for sid, pbox in tracked:
        cheating = False
        pose = sid2pose.get(sid)
        yaw_history.append(pose["yaw"] if pose else 0)
        pitch_history.append(pose["pitch"] if pose else 0)
        avg_yaw = np.mean(np.abs(yaw_history))
        avg_pitch = np.mean(np.abs(pitch_history))

        if auto_calibrated:
            if avg_yaw > YAW_THRESHOLD or avg_pitch > PITCH_THRESHOLD:
                cheating = True
                recorder.log(frame_id, sid, "head_movement")

        if phone_sid.get(sid, False):
            cheating = True
            recorder.log(frame_id, sid, "phone_use")

        labels[sid] = "Cheating" if cheating else "Non-Cheating"

    # --- Draw boxes ---
    for sid, pbox in tracked:
        x1, y1, x2, y2 = map(int, pbox)
        color = (0,0,255) if labels[sid]=="Cheating" else (0,255,0)
        text = labels[sid]
        pose = sid2pose.get(sid)
        if pose:
            text += f" | yaw={pose['yaw']:.1f} pitch={pose['pitch']:.1f}"
        cv2.rectangle(frame_rgb, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame_rgb, text, (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- Streamlit display ---
    video_placeholder.image(frame_rgb, channels="RGB")
    frame_id += 1

cap.release()
