import cv2
import numpy as np
import json

# --- Color Configuration ---
# Use a colormap to assign unique and vibrant colors to players
PLAYER_COLORS = cv2.applyColorMap(np.arange(0, 255, 15, dtype=np.uint8), cv2.COLORMAP_HSV)
BALL_COLOR = (255, 255, 255)  # White
NON_PLAYER_COLOR = (128, 128, 128) # Gray

def draw_annotations(frame, players, ball_position, team_assignments, team_colors=None, player_positions=None):
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

    # --- Draw team convex hulls and defensive block status ---
    if player_positions:
        team_positions_current_frame = {}
        for pid, pos in player_positions.items():
            team_id = team_assignments.get(pid)
            if team_id is not None and team_id != 'non_player':
                team_positions_current_frame.setdefault(team_id, []).append(pos)

        from .tactical_analysis import analyze_team_shape # Local import
        for team_id, positions in team_positions_current_frame.items():
            color = team_colors.get(team_id, (255, 0, 255))

            # Draw Convex Hull for team shape
            if len(positions) > 2:
                points = np.array(positions, dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.polylines(frame, [hull], isClosed=True, color=color, thickness=2)

            # Analyze and display defensive block
            block_status = analyze_team_shape(positions, frame.shape[0])
            if block_status != 'N/A':
                # Display text on the side of the screen
                text_pos_y = 50 if team_id == '0' else 100
                cv2.putText(frame, f"Team {team_id} Block: {block_status}",
                            (frame.shape[1] - 300, text_pos_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

def generate_annotated_video(video_path, output_path, annotations_path):
    """
    Generates an annotated video with tactical drawings.
    """
    with open(annotations_path, 'r') as f:
        analysis_data = json.load(f)

    annotations = analysis_data['annotations_by_frame']
    team_assignments = analysis_data['team_assignments']
    team_colors = {str(k): tuple(v) for k, v in analysis_data['team_colors'].items()}
    frame_skip = analysis_data.get('frame_skip', 1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use 'avc1' or 'mp4v' for H.264 codec, which is broadly compatible
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the correct annotation index based on the frame skip
        annotation_idx = (frame_idx // frame_skip)

        if annotation_idx < len(annotations):
            ann_data = annotations[annotation_idx]

            # Use the 'players' key which now contains box, center, etc.
            players_to_draw = ann_data.get('players', {})
            ball_pos = ann_data.get('ball_pos')

            # The player_positions for shape drawing are now the centers from players_to_draw
            player_positions_for_shape = {pid: p_data['center'] for pid, p_data in players_to_draw.items() if p_data.get('center')}

            annotated_frame = draw_annotations(frame, players_to_draw, ball_pos, team_assignments, team_colors, player_positions_for_shape)
            out.write(annotated_frame)
        else:
            # If no annotation for this frame, write the original frame
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

def create_heatmap(positions, frame_shape):
    """
    Creates a heatmap from a list of positions.
    (This function will be implemented in more detail later)
    """
    print("Generating heatmap...")
    # Logic to create a heatmap from coordinates.
    pass