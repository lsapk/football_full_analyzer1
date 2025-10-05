from ultralytics import YOLO
from utils import load_config
import os

class Detector:
    def __init__(self, model_name=None, device='cpu'):
        cfg = load_config()
        self.model_name = model_name or cfg.get('yolo_model','yolov8n.pt')
        print('Initialising YOLO model:', self.model_name)
        self.model = YOLO(self.model_name)
        self.names = self.model.model.names if hasattr(self.model, 'model') else {}
    def detect(self, source, show=False, classes=None):
        # wrapper that returns ultralytics results iterator (with tracking if requested)
        return self.model.track(source=source, persist=True, device='cpu', show=show, stream=True, classes=classes)

