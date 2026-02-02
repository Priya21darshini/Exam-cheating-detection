import cv2
import time
from detectors import Detector
from tracker import Tracker
from head_pose import HeadPoseEstimator
from recorder import Recorder

detector = Detector()
tracker = Tracker()
pose_estimator = HeadPoseEstimator()
recorder = Recorder()

cap = cv2.VideoCapture(0)
frame_id = 0
labels = {}

# ----- Auto-calibration -----
calibration_time_sec = 5
start_time = time.time()
yaw_values = []
pitch_values = []
auto_calibrated = False

# These will be set dynamically after calibration
YAW_THRESHOLD   = 45
PITCH_THRESHOLD = 45

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) detect (person + phone only)
    detections = detector.detect(frame)
    person_boxes = [d[0] for d in detections if d[1] == 0]
    tracked = tracker.update(person_boxes)

    # 2) pose estimate
    poses = pose_estimator.estimate(frame)

    # 3) assign pose to tracked student (IoU)
    sid2pose = {}
    for sid, pbox in tracked:
        best_iou, best_pose = 0, None
        for pose in poses:
            hbox = pose["bbox"]
            x1=max(pbox[0],hbox[0]); y1=max(pbox[1],hbox[1])
            x2=min(pbox[2],hbox[2]); y2=min(pbox[3],hbox[3])
            inter = max(0,x2-x1)*max(0,y2-y1)
            areaP=(pbox[2]-pbox[0])*(pbox[3]-pbox[1])+1e-6
            areaH=(hbox[2]-hbox[0])*(hbox[3]-hbox[1])+1e-6
            iou=inter/(areaP+areaH-inter)
            if iou>best_iou:
                best_iou,best_pose=iou,pose
        if best_pose and best_iou>0.1:
            sid2pose[sid] = best_pose

    # --- Auto-calibrate normal head pose ---
    elapsed = time.time() - start_time
    if not auto_calibrated and elapsed <= calibration_time_sec:
        for pose in sid2pose.values():
            yaw_values.append(pose["yaw"])
            pitch_values.append(pose["pitch"])
        cv2.putText(frame,"CALIBRATING... Sit normally",(50,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    elif not auto_calibrated:
        # take the max absolute values as normal range and add margin
        max_yaw= max(abs(v) for v in yaw_values) if yaw_values else 0
        max_pitch=max(abs(v) for v in pitch_values) if pitch_values else 0
        YAW_THRESHOLD   = max_yaw   + 10
        PITCH_THRESHOLD = max_pitch + 10
        auto_calibrated = True
        print(f"[CALIB] yaw_th={YAW_THRESHOLD:.1f}  pitch_th={PITCH_THRESHOLD:.1f}")

    # 4) phone detection (per sid)
    phone_sid = {}
    for box,cls in detections:
        if cls == 67:
            cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
            for sid,pbox in tracked:
                if pbox[0]<=cx<=pbox[2] and pbox[1]<=cy<=pbox[3]:
                    phone_sid[sid]=True

    # 5) behavior detection
    for sid, pbox in tracked:
        cheating = False
        behavior_label = ""

        pose = sid2pose.get(sid)
        if pose and auto_calibrated:
            yaw = pose["yaw"]
            pitch = pose["pitch"]

            # Left/right check
            if abs(yaw) > YAW_THRESHOLD:
                cheating = True
                behavior_label = "look_left_right"
                recorder.log(frame_id, sid, behavior_label)

            # Up/down check
            elif abs(pitch) > PITCH_THRESHOLD:
                cheating = True
                behavior_label = "look_up_down"
                recorder.log(frame_id, sid, behavior_label)

        # phone
        if phone_sid.get(sid, False):
            cheating = True
            behavior_label = "phone_use"
            recorder.log(frame_id, sid, behavior_label)

        labels[sid] = "Cheating" if cheating else "Non-Cheating"

    # 6) draw overlay
    for sid,pbox in tracked:
        x1,y1,x2,y2 = map(int,pbox)
        text = labels[sid]
        pose = sid2pose.get(sid)
        if pose:   # also show raw pose values for debug
            text += f" | yaw={pose['yaw']:.1f} pitch={pose['pitch']:.1f}"
        c = (0,0,255) if labels[sid]=="Cheating" else (0,255,0)
        cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
        cv2.putText(frame,text,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1)

    cv2.imshow("Exam Cheating Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
