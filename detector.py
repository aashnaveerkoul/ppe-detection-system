from ultralytics import YOLO
import cv2

class PPEDetector:
    def __init__(self, model_path="best_v2.pt"):
        # Load the pretrained YOLOv8 model
        self.model = YOLO(model_path)
        
    def detect_frame(self, frame, conf=0.4):
        """Runs inference on a single frame/image."""
        results = self.model(frame, conf=conf)[0]
        return results