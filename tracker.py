# Simple tracker placeholder – you can later use SORT or ByteTrack
class Tracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}

    def update(self, detections):
        tracked = []
        for det in detections:
            self.objects[self.next_id] = det
            tracked.append((self.next_id, det))
            self.next_id += 1
        return tracked
