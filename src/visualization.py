import cv2
import numpy as np

# --- Color Configuration ---
# Use a colormap to assign unique and vibrant colors to players
PLAYER_COLORS = cv2.applyColorMap(np.arange(0, 255, 15, dtype=np.uint8), cv2.COLORMAP_HSV)
BALL_COLOR = (255, 255, 255)  # White
TEAM_A_COLOR = (255, 0, 0)   # Blue
TEAM_B_COLOR = (0, 0, 255)   # Red

def draw_annotations(frame, players, ball_position, team_assignments):
    """
    Draws all annotations on a frame of the match.

    Args:
        frame (np.array): The video frame to draw on.
        players (dict): Dictionary containing player information (positions, etc.).
        ball_position (tuple): (x, y) coordinates of the ball.
        team_assignments (dict): Dictionary mapping player IDs to their team ('0' or '1').

    Returns:
        np.array: The frame with annotations.
    """
    if frame is None:
        return None

    # --- Draw trajectories ---
    for pid, player_data in players.items():
        positions_with_frame = player_data.get('positions', [])
        if len(positions_with_frame) > 2:
            # Extract only (x, y) coordinates for drawing, ignoring the frame index
            coord_only_positions = [(int(x), int(y)) for frame_idx, x, y in positions_with_frame]

            # Ensure positions are integers for drawing
            line_points = np.array(coord_only_positions, dtype=np.int32).reshape((-1, 1, 2))
            color = PLAYER_COLORS[pid % len(PLAYER_COLORS)][0].tolist()
            cv2.polylines(frame, [line_points], isClosed=False, color=color, thickness=2)

    # --- Draw player boxes and their IDs ---
    for pid, player_data in players.items():
        last_pos = player_data.get('last_pos')
        if last_pos:
            x, y = int(last_pos[0]), int(last_pos[1])
            team = team_assignments.get(pid)

            # Choose color based on the team
            color = TEAM_A_COLOR if team == '0' else TEAM_B_COLOR if team == '1' else (0, 255, 0)

            # Draw a circle to represent the player
            cv2.circle(frame, (x, y), radius=10, color=color, thickness=-1)
            cv2.putText(frame, str(pid), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # --- Draw the ball ---
    if ball_position:
        # Ensure ball_position is a tuple of integers
        cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), radius=8, color=BALL_COLOR, thickness=-1)

    return frame

def generate_video(input_path, output_path, analysis_data):
    """
    Generates an annotated video from the analysis data.
    (This function will be implemented in more detail later)
    """
    print(f"Generating annotated video to {output_path}...")
    # Logic to read the video, apply draw_annotations frame by frame, and save.
    pass

def create_heatmap(positions, frame_shape):
    """
    Creates a heatmap from a list of positions.
    (This function will be implemented in more detail later)
    """
    print("Generating heatmap...")
    # Logic to create a heatmap from coordinates.
    pass