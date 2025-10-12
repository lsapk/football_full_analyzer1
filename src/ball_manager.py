import numpy as np
from .utils import box_center

class BallManager:
    """
    Manages the state of the ball, including its position, visibility,
    and predicted location when it's not visible.
    """
    def __init__(self, max_unseen_frames=20, min_seen_frames_for_prediction=10):
        self.position = None
        self.predicted_position = None
        self.visible = False
        self.unseen_frames = 0
        self.max_unseen_frames = max_unseen_frames
        self.position_history = []
        self.min_seen_frames_for_prediction = min_seen_frames_for_prediction
        self.last_known_box = None

    def update(self, best_ball):
        """
        Updates the ball's state based on the best detection in the current frame.
        """
        if best_ball:
            self.last_known_box = best_ball['box']
            self.position = box_center(self.last_known_box)
            self.position_history.append(self.position)
            if len(self.position_history) > 30: # Keep history manageable
                self.position_history.pop(0)
            self.visible = True
            self.unseen_frames = 0
            self.predicted_position = None
        else:
            self.visible = False
            self.unseen_frames += 1
            if self.unseen_frames > self.max_unseen_frames:
                self.position = None
                self.position_history = []
                self.last_known_box = None
            else:
                self.predict_next_position()

    def predict_next_position(self):
        """
        Predicts the ball's next position based on its recent movement.
        A simple linear velocity model is used.
        """
        if len(self.position_history) < self.min_seen_frames_for_prediction:
            self.predicted_position = self.position # Hold last known position
            return

        # Calculate average velocity over the last few frames
        velocities = np.diff(np.array(self.position_history), axis=0)
        if velocities.shape[0] > 0:
            avg_velocity = np.mean(velocities[-5:], axis=0)
            self.predicted_position = tuple((np.array(self.position) + avg_velocity).astype(int))
        else:
            self.predicted_position = self.position

    def get_position(self):
        """
        Returns the current or predicted position of the ball.
        """
        return self.position if self.visible else self.predicted_position

    def get_box(self):
        """
        Returns the current or predicted bounding box of the ball.
        """
        if self.visible:
            return self.last_known_box

        if not self.predicted_position or not self.last_known_box:
            return None

        # Create a new box centered at the predicted position, with the same size as the last known box
        w = self.last_known_box[2] - self.last_known_box[0]
        h = self.last_known_box[3] - self.last_known_box[1]
        x1 = self.predicted_position[0] - w / 2
        y1 = self.predicted_position[1] - h / 2
        x2 = self.predicted_position[0] + w / 2
        y2 = self.predicted_position[1] + h / 2
        return [x1, y1, x2, y2]