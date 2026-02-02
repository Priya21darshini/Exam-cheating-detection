from ultralytics import YOLO

# Only these two classes are relevant: 0=person, 67=cell phone (COCO dataset)
PERSON_CLS = 0
PHONE_CLS  = 67

class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model.predict(frame, imgsz=640, conf=0.45, verbose=False)
        output = []
        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                cls = int(cls)
                if cls in (PERSON_CLS, PHONE_CLS):
                    output.append((box.tolist(), cls))
        return output
