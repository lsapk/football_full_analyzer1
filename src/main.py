import os, json, time
import cv2
import numpy as np
import pandas as pd
from .detector import Detector
from .tracker import parse_frame_results
from .utils import box_center, pixel_distance, speed_kmh
from .events import EventManager
from .visualization import draw_annotations
from . import stats
from . import tactical_analysis

def assign_teams_by_clustering(players, initial_positions):
    if len(initial_positions) < 2:
        for i, pid in enumerate(players):
            players[pid]['team'] = str(i)
        return players
    player_ids = list(initial_positions.keys())
    positions = np.array([np.mean(pos, axis=0) for pos in initial_positions.values()])
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    team_assignments = kmeans.fit_predict(positions)
    for i, pid in enumerate(player_ids):
        if pid in players:
            players[pid]['team'] = str(team_assignments[i])
    return players

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

def filter_players(players, min_positions=10):
    return {pid: data for pid, data in players.items() if len(data.get('positions', [])) >= min_positions}

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
    detector = Detector(model_name=model_path)
    event_manager = EventManager(cfg)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_annotated.mp4")
    # Using a more compatible codec for mp4
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps / cfg.get('frame_skip', 1), (width, height))

    # --- Unified Analysis Loop ---
    print("Starting unified analysis loop...")
    results_iter = detector.detect(video_path, show=False)

    players = {}
    team_assignments = {}
    teams_identified = False
    initial_player_positions = {}

    frame_skip = cfg.get('frame_skip', 1)
    frames_to_sample = frame_skip * cfg.get('team_clustering_sample_frames', 10)

    events, team_stats_history = [], []
    team_possession_seconds = {}
    last_owner_pid = None
    current_owner_pid = None

    last_frame_idx = 0
    for frame_idx, res in enumerate(results_iter):
        try:
            if frame_idx % frame_skip != 0: continue
            last_frame_idx = frame_idx
            persons, balls = parse_frame_results(res, detector)
            frame = res.orig_img

            # Dynamically add any new players found by the tracker
            for p in persons:
                pid = p.get('id')
                if pid and pid not in players:
                    players[pid] = {'touches':0,'positions':[],'dist_pixels':0.0,'last_pos':None,'last_frame':None,'max_speed_kmh':0.0,'team':None}

            # Stage 1: Collect positions for team identification
            if not teams_identified:
                for p in persons:
                    pid = p.get('id')
                    if pid: initial_player_positions.setdefault(pid, []).append(box_center(p['box']))

                if frame_idx > frames_to_sample:
                    print("Identifying teams based on collected positions...")
                    players = assign_teams_by_clustering(players, initial_player_positions)
                    team_assignments = {pid: pdata.get('team') for pid, pdata in players.items()}

                    team_ids = set(team_assignments.values())
                    team_possession_seconds = {team_id: 0 for team_id in team_ids if team_id is not None}
                    teams_identified = True
                    print("Teams identified. Continuing full analysis...")

            # Stage 2: Main analysis logic (runs on every frame)
            for p in persons:
                pid = p.get('id')
                if pid and pid in players:
                    p['center'] = box_center(p['box'])
                    stats.update_player_movement(players[pid], p, frame_idx, fps, cfg)

            current_team_stats = {}
            if teams_identified:
                current_team_stats = stats.calculate_team_stats(players, team_assignments, cfg.get('pixels_to_meters'))
                team_stats_history.append(current_team_stats)

            ball = balls[0] if balls else None
            owner = find_ball_owner(ball, persons)
            owner_pid = owner['id'] if owner else None

            if teams_identified and owner_pid and owner_pid in players:
                players[owner_pid]['touches'] += 1
                owner_team = players[owner_pid].get('team')
                if owner_team in team_possession_seconds:
                    team_possession_seconds[owner_team] += (frame_skip / fps)

            new_events = event_manager.update(frame_idx, players, ball, owner_pid)
            events.extend(new_events)

            # Annotation
            ball_pos = box_center(balls[0]['box']) if balls else None
            y_offset = 30
            if teams_identified:
                for team_id, team_data in current_team_stats.items():
                    if team_id is None: continue
                    compactness = team_data.get('compactness', 0)
                    team_name = cfg['team_names'].get(str(team_id), f"Team {team_id}")
                    text = f"{team_name} Compactness: {compactness:.2f}m"
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    y_offset += 30

            annotated_frame = draw_annotations(frame.copy(), players, ball_pos, team_assignments)
            video_writer.write(annotated_frame)
        except Exception as e:
            # print(f"Error in frame {frame_idx}: {e}")
            continue

    video_writer.release()
    print(f"Annotated video saved to {output_video_path}")
    total_duration = last_frame_idx / fps

    # Filter players with too few positions to be considered stable tracks
    min_pos_filter = cfg.get('min_player_positions', 2) # Using the value from main.py config
    print(f"DEBUG: Total players tracked before filtering: {len(players)}")
    players = filter_players(players, min_positions=min_pos_filter)
    print(f"DEBUG: Total players after filtering: {len(players)}")

    print(f"Finished processing. Found {len(players)} stable player tracks.")
    export_results(output_dir, players, events, video_path, cfg, team_possession_seconds, total_duration, team_stats_history, generate_llm_report)