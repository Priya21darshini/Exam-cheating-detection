import csv
import os

class Recorder:
    def __init__(self, path="logs/events.csv"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["frame", "student_id", "behavior"])

    def log(self, frame_id, student_id, behavior):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([frame_id, student_id, behavior])
