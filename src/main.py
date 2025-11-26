import os, json, time
import cv2
import numpy as np
import pandas as pd
from .detector import Detector
from .tracker import parse_frame_results
from .utils import box_center, pixel_distance, speed_kmh, get_dominant_color
from .events import EventManager
from .visualization import draw_annotations
from . import stats
from . import tactical_analysis
from . import database

def convert_keys_to_str_recursive(obj):
    """Recursively converts all keys in a nested dictionary to strings."""
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str_recursive(elem) for elem in obj]
    else:
        return obj

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def assign_teams_by_color(players, player_colors, min_samples=3):
    """
    Assigns teams based on player jersey colors using K-Means clustering.
    This version first calculates the average color for each player, then clusters
    these average colors. This gives each player equal weight in the clustering.
    """
    player_avg_colors = {}
    for pid, colors in player_colors.items():
        if len(colors) >= min_samples:
            player_avg_colors[pid] = np.mean(colors, axis=0)

    if len(player_avg_colors) < 2:
        return {}, {} # Not enough players to form two teams

    # Prepare data for clustering
    pids = list(player_avg_colors.keys())
    avg_colors = np.array(list(player_avg_colors.values()))

    # K-Means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(avg_colors)

    # Map cluster centers to team colors
    team_color_map = {0: tuple(map(int, kmeans.cluster_centers_[0])),
                      1: tuple(map(int, kmeans.cluster_centers_[1]))}

    # Assign teams based on clustering results
    team_assignments = {}
    for i, pid in enumerate(pids):
        team_id = kmeans.labels_[i]
        if pid in players:
            players[pid]['team'] = team_id
            team_assignments[pid] = team_id

    return team_assignments, team_color_map

def find_ball_owner(ball, persons):
    """Finds the player closest to the ball using vectorized operations."""
    if not ball or not persons:
        return None

    ball_center = np.array(box_center(ball['box']))

    # Create a NumPy array of person centers and their corresponding objects
    person_centers = np.array([box_center(p['box']) for p in persons])

    # Calculate distances in a vectorized way
    distances = np.linalg.norm(person_centers - ball_center, axis=1)

    # Find the index of the minimum distance
    min_dist_idx = np.argmin(distances)

    # Return the person object with the minimum distance
    return persons[min_dist_idx]

def filter_players(players, min_positions=10, min_distance_m=20, goalkeeper_ids=None):
    if goalkeeper_ids is None: goalkeeper_ids = []
    non_player_ids = []
    for pid, data in players.items():
        if len(data.get('positions', [])) < min_positions: continue
        is_goalkeeper = pid in goalkeeper_ids
        distance_m = data.get('distance_m', 0)
        if distance_m < min_distance_m and not is_goalkeeper:
            non_player_ids.append(pid)
    return non_player_ids

def format_and_save_results(output_dir, players, events, video_path, cfg, team_possession_seconds, total_time_seconds, team_stats_history):
    """Formats the analysis results into DataFrames and saves them to an SQLite database."""
    team_names = cfg.get('team_names', {"0": "Team A", "1": "Team B"})
    pixels_to_meters = cfg.get('pixels_to_meters', 0.1)
    db_path = os.path.join(output_dir, 'analysis.db')
    video_name = os.path.basename(video_path)

    # --- 1. Events DataFrame ---
    events_df = pd.DataFrame(events)
    if not events_df.empty:
        # Normalize coordinates
        frame_width = cfg.get('frame_width', 1)
        frame_height = cfg.get('frame_height', 1)

        start_pos_df = pd.DataFrame(events_df['start_pos'].tolist(), index=events_df.index)
        end_pos_df = pd.DataFrame(events_df['end_pos'].tolist(), index=events_df.index)

        events_df['start_x'] = (start_pos_df[0] / frame_width) * 105
        events_df['start_y'] = 68 - ((start_pos_df[1] / frame_height) * 68) # Inverser l'axe Y
        events_df['end_x'] = (end_pos_df[0] / frame_width) * 105
        events_df['end_y'] = 68 - ((end_pos_df[1] / frame_height) * 68) # Inverser l'axe Y

        # Map team ID to name
        events_df['team_name'] = events_df['team_id'].map(team_names)
        # Rename and select final columns
        events_df = events_df.rename(columns={'type': 'event_type'})
        events_df = events_df[['frame', 'event_type', 'team_name', 'player_id', 'start_x', 'start_y', 'end_x', 'end_y']]

    # --- 2. Player Statistics DataFrame ---
    player_rows = []
    for pid, d in players.items():
        player_events = events_df[events_df['player_id'] == pid] if not events_df.empty else pd.DataFrame()
        row = {
            'player_id': pid,
            'team_name': team_names.get(str(d.get('team')), f"Team {d.get('team')}"),
            'touches': d.get('touches', 0),
            'passes': player_events[player_events['event_type'] == 'pass'].shape[0] if not player_events.empty else 0,
            'shots': player_events[player_events['event_type'] == 'shot'].shape[0] if not player_events.empty else 0,
            'distance_m': round(d.get('dist_pixels', 0.0) * pixels_to_meters, 2),
            'max_speed_kmh': round(d.get('max_speed_kmh', 0.0), 2)
        }
        player_rows.append(row)
    players_df = pd.DataFrame(player_rows)

    # --- 3. Team Statistics DataFrame ---
    teams_df = pd.DataFrame()
    if not players_df.empty:
        team_summary = players_df.groupby('team_name').agg(
            total_passes=('passes', 'sum'),
            total_shots=('shots', 'sum')
        ).reset_index()

        # Calculate possession
        if total_time_seconds > 0:
            possession_pct = {name: round((team_possession_seconds.get(int(tid), 0) / total_time_seconds) * 100, 2) for tid, name in team_names.items()}
            team_summary['possession_pct'] = team_summary['team_name'].map(possession_pct).fillna(0)

        # Calculate avg_compactness
        if team_stats_history:
            history_df = pd.DataFrame([{'team_name': team_names.get(str(team_id)), **data} for frame_stats in team_stats_history for team_id, data in frame_stats.items()])
            if not history_df.empty:
                avg_compactness = history_df.groupby('team_name')['compactness'].mean().round(2)
                team_summary = team_summary.merge(avg_compactness.rename('avg_compactness'), on='team_name', how='left')

        teams_df = team_summary

    # --- 4. Save to Database ---
    database.save_analysis_to_db(db_path, video_name, players_df, teams_df, events_df)
    print(f"Done. Results saved to {db_path}")

def run_analysis(video_path, output_dir, model_path, config, generate_llm_report=False, progress_callback=None):
    # --- 1. Initial Setup ---
    os.makedirs(output_dir, exist_ok=True)
    cfg = config
    frame_skip = cfg.get('frame_skip', 1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cfg['frame_width'] = width
    cfg['frame_height'] = height
    detector = Detector(model_name=model_path)

    # --- 2. Phase 1: Data Collection ---
    if progress_callback:
        progress_callback(f"Phase 1/3 : Collecte des données de détection (total frames: {total_frames})...")

    results_iter = detector.detect(video_path, show=False)
    all_frame_data = []
    player_color_samples = {}
    initial_player_positions = {}
    last_frame_idx = 0

    if results_iter:
        for frame_idx, res in enumerate(results_iter):
            try:
                if frame_idx % frame_skip != 0: continue

                log_msg = f"Frame {frame_idx}/{total_frames} - Vitesse: {res.speed['preprocess']:.1f}ms pre, {res.speed['inference']:.1f}ms inf, {res.speed['postprocess']:.1f}ms post"
                if progress_callback: progress_callback(log_msg)

                last_frame_idx = frame_idx
                persons, balls = parse_frame_results(res, detector)
                all_frame_data.append({'frame_idx': frame_idx, 'persons': persons, 'balls': balls})

                # Collect color samples and initial positions for better team assignment
                frame_bgr = res.orig_img
                for p in persons:
                    pid = p.get('id')
                    if pid:
                        color = get_dominant_color(frame_bgr, p['box'])
                        if color: player_color_samples.setdefault(pid, []).append(color)
                        initial_player_positions.setdefault(pid, []).append(box_center(p['box']))
            except Exception as e:
                print(f"Erreur lors de la collecte de données à l'image {frame_idx}: {e}")
                continue

    # --- 3. Phase 2: Data Processing & Analysis ---
    if progress_callback: progress_callback("Phase 2/3 : Traitement des données et analyse...")

    # Initialize players dictionary
    players = {}
    for frame_data in all_frame_data:
        for p in frame_data['persons']:
            pid = p.get('id')
            if pid and pid not in players:
                players[pid] = {'touches': 0, 'positions': [], 'dist_pixels': 0.0, 'last_pos': None, 'last_frame': None, 'max_speed_kmh': 0.0, 'team': None, 'last_box': None}

    # Team assignment (more accurate with more samples)
    team_assignments, team_colors = assign_teams_by_color(players, player_color_samples)
    for pid, team_id in team_assignments.items():
        if pid in players: players[pid]['team'] = team_id

    # Process movement, possession, events, and tactical stats
    event_manager = EventManager(cfg)
    events, team_stats_history, team_possession_seconds = [], [], {tid: 0 for tid in team_colors}
    last_owner_pid = None

    for frame_data in all_frame_data:
        frame_idx = frame_data['frame_idx']
        persons = frame_data['persons']
        ball = frame_data['balls'][0] if frame_data['balls'] else None

        for p in persons:
            p['center'] = box_center(p['box'])
            pid = p.get('id')
            if pid and pid in players:
                stats.update_player_movement(players[pid], p, frame_idx, fps, cfg)

        owner = find_ball_owner(ball, persons)
        current_owner_pid = owner['id'] if owner else None
        if current_owner_pid and current_owner_pid in players:
            players[current_owner_pid]['touches'] += 1
            owner_team = players[current_owner_pid].get('team')
            if owner_team in team_possession_seconds:
                team_possession_seconds[owner_team] += (frame_skip / fps)

        new_events = event_manager.update(frame_idx, players, ball, last_owner_pid, current_owner_pid)
        events.extend(new_events)
        last_owner_pid = current_owner_pid

        # Tactical analysis
        team_positions = {team_id: [p['last_pos'] for pid, p in players.items() if p.get('team') == team_id and p.get('last_pos')] for team_id in team_colors}
        frame_team_stats = {}
        for team_id, positions in team_positions.items():
            if len(positions) > 2:
                compactness = tactical_analysis.calculate_compactness(positions, cfg.get('pixels_to_meters', 0.1))
                frame_team_stats[team_id] = {'compactness': compactness}
        if frame_team_stats: team_stats_history.append(frame_team_stats)

    # --- 4. Phase 3: Finalization & Export ---
    if progress_callback: progress_callback("Phase 3/3 : Finalisation et sauvegarde des résultats...")

    # Filter non-players
    goalkeeper_ids = []
    if team_assignments and initial_player_positions:
        team_0_players = {pid: pos for pid, pos in initial_player_positions.items() if players.get(pid, {}).get('team') == 0}
        team_1_players = {pid: pos for pid, pos in initial_player_positions.items() if players.get(pid, {}).get('team') == 1}
        if team_0_players: goalkeeper_ids.append(min(team_0_players.keys(), key=lambda pid: np.mean([p[0] for p in team_0_players[pid]])))
        if team_1_players: goalkeeper_ids.append(max(team_1_players.keys(), key=lambda pid: np.mean([p[0] for p in team_1_players[pid]])))

    non_player_ids = filter_players(players, min_positions=cfg.get('min_player_positions', 10), min_distance_m=cfg.get('min_distance_m_for_player', 20), goalkeeper_ids=goalkeeper_ids)
    for pid in non_player_ids:
        if pid in team_assignments: team_assignments[pid] = 'non_player'

    active_players = {pid: data for pid, data in players.items() if pid not in non_player_ids}

    # Prepare annotations for visualization
    annotations_to_draw = []
    for frame_data in all_frame_data:
        current_annotations = {'players': {}, 'ball_pos': None, 'team_assignments': team_assignments.copy()}
        for p in frame_data['persons']:
            pid = p.get('id')
            if pid in players: current_annotations['players'][pid] = {'last_box': p['box'], 'center': p.get('center')}
        if frame_data['balls']: current_annotations['ball_pos'] = box_center(frame_data['balls'][0]['box'])
        annotations_to_draw.append(current_annotations)

    # Save annotations file
    final_data = {'annotations_by_frame': annotations_to_draw, 'team_assignments': team_assignments, 'team_colors': team_colors, 'non_player_ids': non_player_ids, 'frame_height': height, 'frame_width': width, 'fps': fps, 'frame_skip': frame_skip, 'config': cfg}
    sanitized_data = convert_keys_to_str_recursive(final_data)
    annotations_path = os.path.join(output_dir, 'annotations.json')
    with open(annotations_path, 'w') as f: json.dump(sanitized_data, f, cls=NumpyEncoder)

    if progress_callback: progress_callback(f"Données d'annotation sauvegardées : {annotations_path}")

    # Export statistics
    total_duration = last_frame_idx / fps
    format_and_save_results(output_dir, active_players, events, video_path, cfg, team_possession_seconds, total_duration, team_stats_history)

    if progress_callback: progress_callback(f"Analyse terminée. {len(active_players)} joueurs actifs identifiés.")

    return {"annotations_data": annotations_path, "db_path": os.path.join(output_dir, 'analysis.db'), "video_path": video_path}