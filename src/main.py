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
    all_colors = [color for pid, colors in player_colors.items() for color in colors if color is not None]
    if not all_colors:
        return {}, {}
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(all_colors)
    team_color_map = {0: tuple(map(int, kmeans.cluster_centers_[0])), 1: tuple(map(int, kmeans.cluster_centers_[1]))}
    team_assignments = {}
    for pid, colors in player_colors.items():
        if not colors: continue
        player_avg_color = np.mean(colors, axis=0)
        dist_0 = np.linalg.norm(player_avg_color - kmeans.cluster_centers_[0])
        dist_1 = np.linalg.norm(player_avg_color - kmeans.cluster_centers_[1])
        team_id = 0 if dist_0 < dist_1 else 1
        if pid in players:
            players[pid]['team'] = team_id
            team_assignments[pid] = team_id
    return team_assignments, team_color_map

def find_ball_owner(ball, persons):
    owner = None
    if not ball or not persons: return None
    bx, by = box_center(ball['box'])
    min_dist = float('inf')
    for p in persons:
        cx, cy = box_center(p['box'])
        d = pixel_distance((cx, cy), (bx, by))
        if d < min_dist:
            min_dist, owner = d, p
    return owner

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
    os.makedirs(output_dir, exist_ok=True)
    cfg = config
    frame_skip = cfg.get('frame_skip', 1)
    detector = Detector(model_name=model_path)
    event_manager = EventManager(cfg)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if progress_callback:
        progress_callback(f"Phase 1/2 : Analyse de la détection et du suivi (total frames: {total_frames})...")

    results_iter = detector.detect(video_path, show=False)
    players, team_assignments, team_colors, player_color_samples, initial_player_positions = {}, {}, {}, {}, {}
    events, team_stats_history, team_possession_seconds = [], [], {}
    last_owner_pid = None
    annotations_to_draw = []
    last_frame_idx = 0

    # --- Main Analysis Loop ---
    if results_iter:
        for frame_idx, res in enumerate(results_iter):
            try:
                if frame_idx % frame_skip != 0:
                    continue

                log_msg = f"Frame {frame_idx}/{total_frames} - Vitesse: {res.speed['preprocess']:.1f}ms pre-process, {res.speed['inference']:.1f}ms inférence, {res.speed['postprocess']:.1f}ms post-process"
                if progress_callback:
                    progress_callback(log_msg)

                last_frame_idx = frame_idx
                frame_bgr = res.orig_img
                persons, balls = parse_frame_results(res, detector)

                for p in persons:
                    pid = p.get('id')
                    if pid and pid not in players:
                        players[pid] = {'touches': 0, 'positions': [], 'dist_pixels': 0.0, 'last_pos': None, 'last_frame': None, 'max_speed_kmh': 0.0, 'team': None, 'last_box': None}
            except Exception as e:
                print(f"Erreur à l'image {frame_idx}: {e}")
                continue # Skip corrupted frames

        if not team_assignments:
            for p in persons:
                pid = p.get('id')
                if pid:
                    color = get_dominant_color(frame_bgr, p['box'])
                    if color: player_color_samples.setdefault(pid, []).append(color)
                    initial_player_positions.setdefault(pid, []).append(box_center(p['box']))

            if frame_idx > frame_skip * cfg.get('team_clustering_sample_frames', 20):
                team_assignments, team_colors = assign_teams_by_color(players, player_color_samples)
                if team_assignments:
                    team_ids = set(team_assignments.values())
                    team_possession_seconds = {team_id: 0 for team_id in team_ids if team_id is not None}

        for p in persons:
            p['center'] = box_center(p['box'])
            pid = p.get('id')
            if pid and pid in players: stats.update_player_movement(players[pid], p, frame_idx, fps, cfg)

        ball = balls[0] if balls else None
        owner = find_ball_owner(ball, persons)
        current_owner_pid = owner['id'] if owner else None

        if team_assignments and current_owner_pid and current_owner_pid in players:
            players[current_owner_pid]['touches'] += 1
            owner_team = players[current_owner_pid].get('team')
            if owner_team in team_possession_seconds: team_possession_seconds[owner_team] += (frame_skip / fps)

        new_events = event_manager.update(frame_idx, players, ball, last_owner_pid, current_owner_pid)
        events.extend(new_events)
        last_owner_pid = current_owner_pid

        # Tactical Analysis for Team Stats
        if team_assignments:
            team_positions = {team_id: [] for team_id in set(team_assignments.values()) if team_id != 'non_player'}
            for pid, p_data in players.items():
                team_id = p_data.get('team')
                if team_id in team_positions and p_data.get('last_pos'):
                    team_positions[team_id].append(p_data['last_pos'])

            frame_team_stats = {}
            for team_id, positions in team_positions.items():
                if len(positions) > 2:
                    compactness = tactical_analysis.calculate_compactness(positions, cfg.get('pixels_to_meters', 0.1))
                    frame_team_stats[team_id] = {'compactness': compactness}
            if frame_team_stats:
                team_stats_history.append(frame_team_stats)

        current_annotations = {'players': {}, 'ball_pos': None, 'team_assignments': team_assignments.copy()}
        for p in persons:
            pid = p.get('id')
            if pid in players: current_annotations['players'][pid] = {'last_box': p['box'], 'center': p.get('center')}
        if balls: current_annotations['ball_pos'] = box_center(balls[0]['box'])
        annotations_to_draw.append(current_annotations)

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

    if progress_callback:
        progress_callback("Phase 2/2 : Sauvegarde des données d'analyse...")

    final_data = {'annotations_by_frame': annotations_to_draw, 'team_assignments': team_assignments, 'team_colors': team_colors, 'non_player_ids': non_player_ids, 'frame_height': height, 'frame_width': width, 'fps': fps, 'frame_skip': frame_skip, 'config': cfg}
    sanitized_data = convert_keys_to_str_recursive(final_data)
    annotations_path = os.path.join(output_dir, 'annotations.json')
    with open(annotations_path, 'w') as f:
        json.dump(sanitized_data, f, cls=NumpyEncoder)

    if progress_callback:
        progress_callback(f"Données d'annotation sauvegardées : {annotations_path}")

    total_duration = last_frame_idx / fps
    if progress_callback:
        progress_callback(f"Analyse terminée. {len(active_players)} joueurs suivis avec succès.")

    export_results(output_dir, active_players, events, video_path, cfg, team_possession_seconds, total_duration, team_stats_history, generate_llm_report)

    return {"annotations_data": annotations_path, "player_stats": os.path.join(output_dir, 'players_stats.csv'), "team_stats": os.path.join(output_dir, 'team_stats.csv'), "events": os.path.join(output_dir, 'events.csv'), "video_path": video_path}