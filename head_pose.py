import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# FaceMesh landmark indices for rough head pose
LM = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291
}

MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],        # nose
    [0.0, -63.6, -12.5],    # chin
    [-43.3, 32.7, -26.0],   # left eye
    [43.3, 32.7, -26.0],    # right eye
    [-28.9, -28.9, -24.1],  # left mouth
    [28.9, -28.9, -24.1]    # right mouth
], dtype=np.float32)


class HeadPoseEstimator:
    def __init__(self):
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True
        )

    def estimate(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        outputs = []
        if res.multi_face_landmarks:
            for face_lms in res.multi_face_landmarks:
                lms = face_lms.landmark

                # 2D image points
                img_pts = np.array([
                    (lms[LM["nose"]].x * w,  lms[LM["nose"]].y * h),
                    (lms[LM["chin"]].x * w,  lms[LM["chin"]].y * h),
                    (lms[LM["left_eye"]].x * w,  lms[LM["left_eye"]].y * h),
                    (lms[LM["right_eye"]].x * w, lms[LM["right_eye"]].y * h),
                    (lms[LM["left_mouth"]].x * w,  lms[LM["left_mouth"]].y * h),
                    (lms[LM["right_mouth"]].x * w, lms[LM["right_mouth"]].y * h)
                ], dtype=np.float32)

                # Camera matrix
                focal_length = w
                cam_matrix = np.array([
                    [focal_length, 0, w/2.0],
                    [0, focal_length, h/2.0],
                    [0, 0, 1]
                ], dtype=np.float32)

                dist = np.zeros((4, 1))
                ok, rvec, tvec = cv2.solvePnP(
                    MODEL_POINTS_3D, img_pts, cam_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    continue

                rot_matrix, _ = cv2.Rodrigues(rvec)
                proj = cv2.hconcat((rot_matrix, tvec))
                _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
                yaw, pitch, roll = euler.flatten()

                # Build head bounding box
                xs = [lm.x for lm in lms]
                ys = [lm.y for lm in lms]
                x1 = int(min(xs) * w)
                y1 = int(min(ys) * h)
                x2 = int(max(xs) * w)
                y2 = int(max(ys) * h)

                outputs.append({
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "roll": float(roll),
                    "bbox": [x1, y1, x2, y2]
                })
        return outputs
