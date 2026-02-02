import cv2
from .utils import put_text

def draw_labels(frame, tracked_objs, labels):
    for tid, bbox in tracked_objs:
        x1, y1, x2, y2 = map(int, bbox)
        label = labels.get(tid, "Non-Cheating")
        color = (0,0,255) if label == "Cheating" else (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        put_text(frame, f"ID:{tid} {label}", (x1, y1-10), color)
    return frame
