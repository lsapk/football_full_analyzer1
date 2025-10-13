# import cv2
# import numpy as np

# --- Color Configuration ---
# Use a colormap to assign unique and vibrant colors to players
# PLAYER_COLORS = cv2.applyColorMap(np.arange(0, 255, 15, dtype=np.uint8), cv2.COLORMAP_HSV)
BALL_COLOR = (255, 255, 255)  # White
NON_PLAYER_COLOR = (128, 128, 128) # Gray

def draw_annotations(frame, players, ball_position, team_assignments, team_colors=None, player_positions=None):
    import cv2
    import numpy as np
    """
    Draws all annotations on a frame of the match.

    Args:
        frame (np.array): The video frame to draw on.
        players (dict): Dictionary containing player information (boxes, etc.).
        ball_position (tuple): (x, y) coordinates of the ball.
        team_assignments (dict): Dictionary mapping player IDs to their team ('0' or '1').
        team_colors (dict): Dictionary mapping team IDs ('0', '1') to their BGR color.
        player_positions (dict): Dictionary mapping player IDs to their (x,y) center for the CURRENT frame.

    Returns:
        np.array: The frame with annotations.
    """
    if team_colors is None:
        team_colors = {'0': (0, 255, 0), '1': (0, 0, 255)} # Default to Green/Red
    if frame is None:
        return None

    # --- Draw trajectories (COMMENTED OUT) ---
    # for pid, player_data in players.items():
    #     positions_with_frame = player_data.get('positions', [])
    #     if len(positions_with_frame) > 2:
    #         # Extract only (x, y) coordinates for drawing, ignoring the frame index
    #         coord_only_positions = [(int(x), int(y)) for frame_idx, x, y in positions_with_frame]

    #         # Ensure positions are integers for drawing
    #         line_points = np.array(coord_only_positions, dtype=np.int32).reshape((-1, 1, 2))
    #         color = PLAYER_COLORS[pid % len(PLAYER_COLORS)][0].tolist()
    #         cv2.polylines(frame, [line_points], isClosed=False, color=color, thickness=2)

    # --- Draw player boxes and their IDs ---
    for pid, player_data in players.items():
        last_box = player_data.get('last_box')
        if last_box is not None:
            x1, y1, x2, y2 = map(int, last_box)
            team = team_assignments.get(pid)

            if team == 'non_player':
                color = NON_PLAYER_COLOR
            else:
                # Get color from the team_colors dict, with a fallback
                color = team_colors.get(team, (255, 0, 255)) # Default to magenta if team ID not in dict

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text (ID and Number)
            label = f"ID: {pid}"
            if player_data.get('number') is not None:
                label += f" N: {player_data['number']}"

            # Put label above the box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Draw the ball ---
    if ball_position:
        # Ensure ball_position is a tuple of integers
        cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), radius=8, color=BALL_COLOR, thickness=-1)

    # --- Draw team convex hulls using current frame positions ---
    if player_positions:
        team_positions_current_frame = {}
        for pid, pos in player_positions.items():
            team_id = team_assignments.get(pid)
            if team_id is not None and team_id != 'non_player':
                team_positions_current_frame.setdefault(team_id, []).append(pos)

        for team_id, positions in team_positions_current_frame.items():
            if len(positions) > 2:
                points = np.array(positions, dtype=np.int32)
                hull = cv2.convexHull(points)
                color = team_colors.get(team_id, (255, 0, 255))
                cv2.polylines(frame, [hull], isClosed=True, color=color, thickness=2)

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