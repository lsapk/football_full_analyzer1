import os
import requests
from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, model_name):
        """
        Initializes the detector with a specific YOLO model.

        Args:
            model_name (str): The path to the YOLO model file (e.g., 'models/yolov8n.pt').
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing YOLO model: {self.model_name} on device: {self.device}")

        # Check if the model file exists and download it if it doesn't
        if not os.path.isfile(self.model_name):
            print(f"Model file not found at {self.model_name}. Downloading...")
            self._download_model()

        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        # Updated to access class names directly from the model object as per recent ultralytics versions
        self.names = self.model.names

    def _download_model(self):
        """
        Downloads the model file from a hardcoded URL.
        """
        url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-obb.pt"

        # Ensure the 'models' directory exists
        os.makedirs(os.path.dirname(self.model_name), exist_ok=True)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(self.model_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded successfully and saved to {self.model_name}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the model: {e}")
            raise

    def detect(self, source, show=False, classes=None):
        """
        Runs object detection and tracking on a video source with performance optimizations.
        """
        # Performance optimizations: half-precision and smaller image size
        return self.model.track(
            source=source,
            tracker='football-tracker.yaml',
            persist=True,
            device=self.device,
            show=show,
            stream=True,
            classes=classes,
            half=True,       # Use FP16 for faster inference
            imgsz=640        # Resize input frames for faster processing
        )