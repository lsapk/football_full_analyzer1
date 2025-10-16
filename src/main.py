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

def assign_teams_by_color(players, player_colors, min_samples=3):
    """
    Assigns teams to players based on the dominant color of their jerseys.

    Returns:
        tuple: (team_assignments, team_colors)
    """
    all_colors = [color for pid, colors in player_colors.items() for color in colors if color is not None]
    if not all_colors:
        return {}, {}

    # Cluster all collected colors into two main team colors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(all_colors)
    team_color_map = {0: tuple(map(int, kmeans.cluster_centers_[0])), 1: tuple(map(int, kmeans.cluster_centers_[1]))}

    team_assignments = {}
    for pid, colors in player_colors.items():
        if not colors:
            continue
        # Assign team based on the closest team color
        player_avg_color = np.mean(colors, axis=0)
        dist_0 = np.linalg.norm(player_avg_color - kmeans.cluster_centers_[0])
        dist_1 = np.linalg.norm(player_avg_color - kmeans.cluster_centers_[1])
        team_id = '0' if dist_0 < dist_1 else '1'
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
    if goalkeeper_ids is None:
        goalkeeper_ids = []

    non_player_ids = []
    for pid, data in players.items():
        if len(data.get('positions', [])) < min_positions:
            continue

        is_goalkeeper = pid in goalkeeper_ids
        distance_m = data.get('distance_m', 0)

        if distance_m < min_distance_m and not is_goalkeeper:
            non_player_ids.append(pid)

    return non_player_ids

def export_results(output_dir, players, events, video_path, cfg, team_possession_seconds, total_time_seconds, team_stats_history, generate_llm_report=False):
    team_names = cfg.get('team_names', {})

    # --- Export Player Stats ---
    player_rows = []
    for pid, d in players.items():
        team_id = d.get('team')
        player_rows.append({
            'player_id': pid, 'number': d.get('number'), 'team_id': team_id,
            'team_name': team_names.get(str(team_id), f'Team {team_id}'), 'touches': d.get('touches'),
            'distance_m': round(d.get('dist_pixels', 0.0) * cfg.get('pixels_to_meters', 0.1), 2),
            'max_speed_kmh': round(d.get('max_speed_kmh', 0.0), 2)
        })
    player_df = pd.DataFrame(player_rows)
    player_stats_path = os.path.join(output_dir, 'players_stats.csv')
    player_df.to_csv(player_stats_path, index=False)

    # --- Export Events ---
    events_df = pd.DataFrame(events) if events else pd.DataFrame()
    events_path = os.path.join(output_dir, 'events.csv')
    if not events_df.empty:
        events_df.to_csv(events_path, index=False)

    # --- Calculate and Export Team Stats ---
    team_stats_df = pd.DataFrame()
    if not player_df.empty and 'team_id' in player_df.columns:
        history_df = pd.DataFrame([
            {'team_id': team_id, **data} for frame_stats in team_stats_history for team_id, data in frame_stats.items()
        ])

        if not history_df.empty:
            avg_compactness = history_df.groupby('team_id')['compactness'].mean().round(2)
            team_stats_df = pd.DataFrame(index=avg_compactness.index)
            team_stats_df['avg_compactness_m'] = avg_compactness

            possession_series = pd.Series(team_possession_seconds, name='possession_seconds')
            if total_time_seconds > 0:
                team_stats_df['possession_pct'] = round((possession_series / total_time_seconds) * 100, 2)

            team_stats_df['total_distance_m'] = player_df.groupby('team_id')['distance_m'].sum()
            if not events_df.empty:
                event_counts = events_df.groupby(['team_id', 'type']).size().unstack(fill_value=0)
                event_counts.columns = [f"{col}s" for col in event_counts.columns]
                team_stats_df = team_stats_df.join(event_counts)

            team_stats_df.fillna(0, inplace=True)
            team_stats_df['team_name'] = team_stats_df.index.map(lambda x: team_names.get(str(x), f"Team {x}"))
            team_stats_path = os.path.join(output_dir, 'team_stats.csv')
            team_stats_df.to_csv(team_stats_path, index_label='team_id')

    # --- Generate LLM Report (if enabled) ---
    if generate_llm_report:
        if not team_stats_df.empty and not player_df.empty:
            report = tactical_analysis.generate_tactical_report(team_stats_df, player_df, events_df)
            report_path = os.path.join(output_dir, 'tactical_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Tactical report saved to {report_path}")
        else:
            print("Skipping LLM report generation due to missing stats.")

    # --- Export Summary ---
    summary = {'video': os.path.basename(video_path), 'n_players': len(players), 'n_events': len(events)}
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Results saved in {output_dir}")

def run_analysis(video_path, output_dir, model_path, config, generate_llm_report=False):
    os.makedirs(output_dir, exist_ok=True)
    cfg = config
    frame_skip = cfg.get('frame_skip', 1)
    detector = Detector(model_name=model_path)
    event_manager = EventManager(cfg)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    yield f"Phase 1/2 : Analyse de la détection et du suivi (total frames: {total_frames})..."
    results_iter = detector.detect(video_path, show=False, verbose=False)

    players = {}
    team_assignments = {}
    team_colors = {}
    teams_identified = False
    player_color_samples = {}
    initial_player_positions = {} # Re-add for goalkeeper detection
    frame_skip = cfg.get('frame_skip', 1)
    frames_to_sample = frame_skip * cfg.get('team_clustering_sample_frames', 20)
    events, team_stats_history = [], []
    team_possession_seconds = {}
    last_owner_pid = None
    annotations_to_draw = []

    last_frame_idx = 0
    for frame_idx, res in enumerate(results_iter):
        if frame_idx % frame_skip != 0:
            continue

        log_msg = f"Frame {frame_idx}/{total_frames} - {res.speed['preprocess']:.1f}ms pre-process, {res.speed['inference']:.1f}ms inference, {res.speed['postprocess']:.1f}ms post-process"
        yield log_msg

        last_frame_idx = frame_idx
        frame_bgr = res.orig_img
        persons, balls = parse_frame_results(res, detector)

        for p in persons:
            pid = p.get('id')
            if pid and pid not in players:
                players[pid] = {'touches': 0, 'positions': [], 'dist_pixels': 0.0, 'last_pos': None, 'last_frame': None, 'max_speed_kmh': 0.0, 'team': None, 'last_box': None}

        if not teams_identified:
            for p in persons:
                pid = p.get('id')
                if pid:
                    # Collect color samples
                    color = get_dominant_color(frame_bgr, p['box'])
                    if color:
                        player_color_samples.setdefault(pid, []).append(color)
                    # Collect initial positions for goalkeeper detection
                    initial_player_positions.setdefault(pid, []).append(box_center(p['box']))

            if frame_idx > frames_to_sample:
                team_assignments, team_colors = assign_teams_by_color(players, player_color_samples)
                if team_assignments:
                    team_ids = set(team_assignments.values())
                    team_possession_seconds = {team_id: 0 for team_id in team_ids if team_id is not None}
                    teams_identified = True
                    print("Teams identified by color:", team_colors)

        for p in persons:
            p['center'] = box_center(p['box'])
            pid = p.get('id')
            if pid and pid in players:
                stats.update_player_movement(players[pid], p, frame_idx, fps, cfg)

        ball = balls[0] if balls else None
        owner = find_ball_owner(ball, persons)
        current_owner_pid = owner['id'] if owner else None
        if teams_identified and current_owner_pid and current_owner_pid in players:
            players[current_owner_pid]['touches'] += 1
            owner_team = players[current_owner_pid].get('team')
            if owner_team in team_possession_seconds:
                team_possession_seconds[owner_team] += (frame_skip / fps)

        new_events = event_manager.update(frame_idx, players, ball, last_owner_pid, current_owner_pid)
        events.extend(new_events)
        last_owner_pid = current_owner_pid

        # Store annotations for this frame
        current_annotations = {'players': {}, 'ball_pos': None, 'team_assignments': team_assignments.copy()}
        for p in persons:
            pid = p.get('id')
            if pid in players:
                # Add both box and center position for this frame
                current_annotations['players'][pid] = {'last_box': p['box'], 'center': p.get('center')}
        if balls:
            current_annotations['ball_pos'] = box_center(balls[0]['box'])
        annotations_to_draw.append(current_annotations)

    # --- Post-Analysis Filtering ---
    goalkeeper_ids = []
    if teams_identified and initial_player_positions:
        team_0_players = {pid: pos for pid, pos in initial_player_positions.items() if players.get(pid, {}).get('team') == '0'}
        team_1_players = {pid: pos for pid, pos in initial_player_positions.items() if players.get(pid, {}).get('team') == '1'}
        if team_0_players:
            gk_0 = min(team_0_players.keys(), key=lambda pid: np.mean([p[0] for p in team_0_players[pid]]))
            goalkeeper_ids.append(gk_0)
        if team_1_players:
            gk_1 = max(team_1_players.keys(), key=lambda pid: np.mean([p[0] for p in team_1_players[pid]]))
            goalkeeper_ids.append(gk_1)

    non_player_ids = filter_players(
        players,
        min_positions=cfg.get('min_player_positions', 10),
        min_distance_m=cfg.get('min_distance_m_for_player', 20),
        goalkeeper_ids=goalkeeper_ids
    )
    for pid in non_player_ids:
        if pid in team_assignments:
            team_assignments[pid] = 'non_player'

    active_players = {pid: data for pid, data in players.items() if pid not in non_player_ids}

    # --- Save analysis data for interactive player ---
    yield "Phase 2/2 : Sauvegarde des données d'analyse..."

    # Convert all numpy int keys to string for JSON compatibility
    # This needs to be done on the final version of the dictionaries
    final_team_assignments = {str(k): v for k, v in team_assignments.items()}
    final_team_colors = {str(k): v for k, v in team_colors.items()}

    for frame_data in annotations_to_draw:
        if 'players' in frame_data and frame_data['players']:
            frame_data['players'] = {str(k): v for k, v in frame_data['players'].items()}
        # IMPORTANT: Ensure the team_assignments inside each frame is also string-keyed
        if 'team_assignments' in frame_data:
             frame_data['team_assignments'] = {str(k): v for k, v in frame_data['team_assignments'].items()}


    final_data = {
        'annotations_by_frame': annotations_to_draw,
        'team_assignments': final_team_assignments,
        'team_colors': final_team_colors,
        'non_player_ids': non_player_ids,
        'frame_height': height,
        'frame_width': width,
        'fps': fps,
        'frame_skip': frame_skip,
        'config': cfg
    }

    annotations_path = os.path.join(output_dir, 'annotations.json')
    with open(annotations_path, 'w') as f:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        json.dump(final_data, f, cls=NumpyEncoder)

    yield f"Données d'annotation sauvegardées : {annotations_path}"

    total_duration = last_frame_idx / fps
    yield f"Analyse terminée. {len(active_players)} joueurs suivis avec succès."
    export_results(output_dir, active_players, events, video_path, cfg, team_possession_seconds, total_duration, team_stats_history, generate_llm_report)

    # Yield the final results dictionary
    yield {
        "annotations_data": annotations_path,
        "player_stats": os.path.join(output_dir, 'players_stats.csv'),
        "team_stats": os.path.join(output_dir, 'team_stats.csv'),
        "events": os.path.join(output_dir, 'events.csv'),
        "video_path": video_path
    }