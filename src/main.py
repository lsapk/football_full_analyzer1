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

def export_results(output_dir, players, events, video_path, cfg, team_possession_seconds, total_time_seconds, team_stats_history, generate_llm_report=False):
    team_names = cfg.get('team_names', {"0": "Team A", "1": "Team B"})
    pixels_to_meters = cfg.get('pixels_to_meters', 0.1)

    # --- Player Statistics ---
    player_rows = []
    events_df = pd.DataFrame(events) if events else pd.DataFrame(columns=['type', 'player_id'])

    for pid, d in players.items():
        player_events = events_df[events_df['player_id'] == pid]
        row = {
            'ID Joueur': pid,
            'Numéro': d.get('number', 'N/A'),
            'Équipe': team_names.get(str(d.get('team')), f"Équipe {d.get('team')}"),
            'Touches': d.get('touches', 0),
            'Passes': player_events[player_events['type'] == 'PASS'].shape[0],
            'Tirs': player_events[player_events['type'] == 'SHOT'].shape[0],
            'Distance (m)': round(d.get('dist_pixels', 0.0) * pixels_to_meters, 2),
            'Vitesse Max (km/h)': round(d.get('max_speed_kmh', 0.0), 2)
        }
        player_rows.append(row)

    player_df = pd.DataFrame(player_rows)
    player_stats_path = os.path.join(output_dir, 'players_stats.csv')
    player_df.to_csv(player_stats_path, index=False)

    # --- Team Statistics ---
    team_stats_df = pd.DataFrame()
    if not player_df.empty and 'Équipe' in player_df.columns:
        # Use team names for grouping
        team_df = player_df.groupby('Équipe').agg({
            'Touches': 'sum',
            'Passes': 'sum',
            'Tirs': 'sum',
            'Distance (m)': 'sum'
        })

        # Calculate possession
        if total_time_seconds > 0:
            possession_pct = {team_names.get(str(k), f"Équipe {k}"): round((v / total_time_seconds) * 100, 2) for k, v in team_possession_seconds.items()}
            team_df['Possession (%)'] = team_df.index.map(possession_pct).fillna(0)

        # Calculate avg_compactness from history
        if team_stats_history:
            # Map team_id to team_name before calculating stats
            history_df = pd.DataFrame([{'team_name': team_names.get(str(team_id)), **data}
                                       for frame_stats in team_stats_history
                                       for team_id, data in frame_stats.items()])
            if not history_df.empty:
                avg_compactness = history_df.groupby('team_name')['compactness'].mean().round(2)
                team_df['Compacité Moyenne (m)'] = avg_compactness

        team_stats_df = team_df.reset_index().rename(columns={'index': 'Équipe'})

    team_stats_path = os.path.join(output_dir, 'team_stats.csv')
    team_stats_df.to_csv(team_stats_path, index=False)
    if generate_llm_report:
        if not team_stats_df.empty and not player_df.empty:
            report = tactical_analysis.generate_tactical_report(team_stats_df, player_df, events_df)
            report_path = os.path.join(output_dir, 'tactical_report.txt')
            with open(report_path, 'w') as f: f.write(report)
            print(f"Tactical report saved to {report_path}")
        else: print("Skipping LLM report generation due to missing stats.")
    summary = {'video': os.path.basename(video_path), 'n_players': len(players), 'n_events': len(events)}
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f: json.dump(summary, f, indent=2)
    print(f"Done. Results saved in {output_dir}")

def run_analysis(video_path, output_dir, model_path, config, generate_llm_report=False, progress_callback=None):
    # --- 1. Initial Setup ---
    os.makedirs(output_dir, exist_ok=True)
    cfg = config
    frame_skip = cfg.get('frame_skip', 1)
    detector = Detector(model_name=model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

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
    export_results(output_dir, active_players, events, video_path, cfg, team_possession_seconds, total_duration, team_stats_history, generate_llm_report)

    if progress_callback: progress_callback(f"Analyse terminée. {len(active_players)} joueurs actifs identifiés.")

    return {"annotations_data": annotations_path, "player_stats": os.path.join(output_dir, 'players_stats.csv'), "team_stats": os.path.join(output_dir, 'team_stats.csv'), "events": os.path.join(output_dir, 'events.csv'), "video_path": video_path}