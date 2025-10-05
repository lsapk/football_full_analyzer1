from ultralytics import YOLO
from utils import load_config
import os
import torch

class Detector:
    def __init__(self, model_name=None):
        cfg = load_config()
        self.model_name = model_name or cfg.get('yolo_model','yolov8n.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initialising YOLO model: {self.model_name} on device: {self.device}")
        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        self.names = self.model.model.names if hasattr(self.model, 'model') else {}

    def detect(self, source, show=False, classes=None):
        return self.model.track(source=source, tracker='bytetrack.yaml', persist=True, device=self.device, show=show, stream=True, classes=classes)

