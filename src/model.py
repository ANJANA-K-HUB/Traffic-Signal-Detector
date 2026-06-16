import os
from ultralytics import YOLO

class TrafficModel:
    def __init__(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Custom weights not found at {weights_path}. Please download best.pt from Colab.")
        self.model = YOLO(weights_path)
        print(f" Successfully loaded custom model from {weights_path}")

    def predict(self, frame, imgsz: int, conf: float):
        return self.model(frame, imgsz=imgsz, conf=conf, verbose=False)