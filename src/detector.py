import os
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

        # Check if the model file exists
        if not os.path.isfile(self.model_name):
             raise FileNotFoundError(f"Model file not found at {self.model_name}")

        self.model = YOLO(self.model_name)
        self.model.to(self.device)
        self.names = self.model.model.names if hasattr(self.model, 'model') else {}

    def detect(self, source, show=False, classes=None):
        """
        Runs the object detection and tracking on a video source.

        Args:
            source (str): Path to the video file.
            show (bool): If True, displays the video with annotations.
            classes (list): A list of class IDs to filter for (e.g., [0] for persons).

        Returns:
            An iterator for the tracking results.
        """
        # Using 'botsort.yaml' as a robust, standard tracker.
        return self.model.track(
            source=source,
            tracker='botsort.yaml',
            persist=True,
            device=self.device,
            show=show,
            stream=True,
            classes=classes
        )