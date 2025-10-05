import argparse, os, json, time
import cv2, numpy as np, pandas as pd
from detector import Detector
from tracker import parse_frame_results
from utils import load_config, box_center, pixel_distance, speed_kmh, get_player_team
from events import detect_pass
from ocr_numero import JerseyOCR

def find_ball_owner(ball, persons):
    owner = None
    if not ball or not persons:
        return None

    bx, by = box_center(ball['box'])
    min_dist = float('inf')

    for p in persons:
        cx, cy = box_center(p['box'])
        d = pixel_distance((cx, cy), (bx, by))
        if d < min_dist:
            min_dist = d
            owner = p
    return owner

def update_player_stats(players, pid, owner_obj, frame_idx, fps, cfg, frame=None, ocr_instance=None):
    players.setdefault(pid, {'touches':0,'positions':[],'dist_pixels':0.0,'last_pos':None,'last_frame':None,'max_speed_kmh':0.0,'number':None, 'team':None})
    players[pid]['touches'] += 1
    cx, cy = box_center(owner_obj['box'])
    players[pid]['positions'].append((frame_idx, cx, cy))

    if players[pid]['last_pos'] is not None:
        dd = pixel_distance(players[pid]['last_pos'], (cx, cy))
        players[pid]['dist_pixels'] += dd
        dt = (frame_idx - players[pid]['last_frame']) / fps if players[pid]['last_frame'] is not None else 1.0/fps
        sp = speed_kmh(dd, dt if dt > 0 else 1.0/fps, cfg.get('pixels_to_meters', 0.01))
        if sp > players[pid]['max_speed_kmh']:
            players[pid]['max_speed_kmh'] = sp

    players[pid]['last_pos'] = (cx, cy)
    players[pid]['last_frame'] = frame_idx

    if frame is not None and ocr_instance and len(players[pid]['positions']) % 50 == 1:
        try:
            num = ocr_instance.read_number(frame, owner_obj['box'])
            if num:
                players[pid]['number'] = num
        except Exception:
            pass

def export_results(output_dir, players, events, video_path, cfg, team_possession_seconds, total_time_seconds):
    team_names = cfg.get('team_names', {})

    # Player stats
    player_rows = []
    for pid, d in players.items():
        team_id = d.get('team')
        player_rows.append({
            'player_id': pid, 'number': d.get('number'), 'team_id': team_id,
            'team_name': team_names.get(str(team_id), 'Unknown'), 'touches': d.get('touches'),
            'distance_m': round(d.get('dist_pixels', 0.0) * cfg.get('pixels_to_meters', 0.01), 2),
            'max_speed_kmh': round(d.get('max_speed_kmh', 0.0), 2)
        })
    player_df = pd.DataFrame(player_rows)
    player_df.to_csv(os.path.join(output_dir, 'players_stats.csv'), index=False)

    # Event stats
    pd.DataFrame(events).to_csv(os.path.join(output_dir, 'events.csv'), index=False)

    # Team stats
    team_stats = {k: {'possession_pct': 0, 'passes': 0, 'distance_m': 0} for k in team_names.keys()}
    for team_id, possession_time in team_possession_seconds.items():
        if total_time_seconds > 0:
            team_stats[team_id]['possession_pct'] = round((possession_time / total_time_seconds) * 100, 2)

    if 'team_id' in player_df.columns:
        team_dist = player_df.groupby('team_id')['distance_m'].sum()
        for team_id, total_dist in team_dist.items():
            if str(team_id) in team_stats:
                team_stats[str(team_id)]['distance_m'] = round(total_dist, 2)

    for event in events:
        if event.get('type') == 'pass':
            team_id = event.get('team_from')
            if team_id in team_stats:
                team_stats[team_id]['passes'] += 1

    team_rows = []
    for team_id, stats in team_stats.items():
        row = {'team_id': team_id, 'team_name': team_names.get(team_id, 'Unknown')}
        row.update(stats)
        team_rows.append(row)
    pd.DataFrame(team_rows).to_csv(os.path.join(output_dir, 'team_stats.csv'), index=False)

    # Summary
    summary = {'video': video_path, 'n_players': len(players), 'n_events': len(events)}
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Done. Results saved in', output_dir)

def main(args):
    os.makedirs(args.output, exist_ok=True)
    cfg = load_config(args.config)

    detector = Detector(model_name=cfg.get('yolo_model', 'yolov8n.pt'))
    ocr = JerseyOCR(langs=['en'])
    results_iter = detector.detect(args.video, show=False)

    players, events = {}, []
    ball_tracks = {}
    last_owner, last_ball_time = None, 0
    team_possession_seconds = {k: 0 for k in cfg.get('team_names', {}).keys()}

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    frame_skip = cfg.get('frame_skip', 1)
    total_processed_frames = 0
    last_frame_idx = 0

    print(f"Processing video with frame_skip={frame_skip}. Analyzing 1 frame every {frame_skip} frames.")
    print('Starting processing...')
    for frame_idx, res in enumerate(results_iter):
        try:
            if frame_idx % frame_skip != 0:
                continue

            last_frame_idx = frame_idx
            total_processed_frames += 1
            frame_time = frame_idx / fps
            persons, balls = parse_frame_results(res, detector)
            frame = res.orig_img if hasattr(res, 'orig_img') else None

            if frame is not None:
                for p in persons:
                    pid = p['id']
                    if pid and players.get(pid, {}).get('team') is None:
                        team = get_player_team(frame, p['box'], cfg.get('team_a_colors', []), cfg.get('team_b_colors', []))
                        if team:
                            players.setdefault(pid, {'touches':0,'positions':[],'dist_pixels':0.0,'last_pos':None,'last_frame':None,'max_speed_kmh':0.0,'number':None, 'team':None})
                            players[pid]['team'] = team

            ball = balls[0] if balls else None
            if ball:
                bid = ball.get('id')
                if bid:
                    ball_tracks.setdefault(bid, []).append((frame_idx, *box_center(ball['box'])))
                last_ball_time = frame_time

            owner = find_ball_owner(ball, persons)
            if owner:
                pid = owner['id']
                if pid:
                    update_player_stats(players, pid, owner, frame_idx, fps, cfg, frame, ocr)
                    owner_team = players.get(pid, {}).get('team')
                    if owner_team in team_possession_seconds:
                        team_possession_seconds[owner_team] += (frame_skip / fps)

                if last_owner and pid != last_owner:
                    tdiff = frame_time - last_ball_time
                    prev_pos = players.get(last_owner, {}).get('last_pos')
                    curr_pos = box_center(owner['box'])
                    if prev_pos and detect_pass(last_owner, pid, prev_pos, curr_pos, tdiff, cfg):
                        pass_event = {'type':'pass','time_s':frame_time,'from':last_owner,'to':pid}
                        p_from, p_to = players.get(last_owner), players.get(pid)
                        if p_from and p_to:
                            pass_event['team_from'], pass_event['team_to'] = p_from.get('team'), p_to.get('team')
                        events.append(pass_event)
                last_owner = pid
        except Exception as e:
            print(f"Skipping frame {frame_idx} due to error: {e}")
            continue

    total_duration = last_frame_idx / fps
    export_results(args.output, players, events, args.video, cfg, team_possession_seconds, total_duration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', default='results')
    parser.add_argument('--config', default='config.json')
    args = parser.parse_args()
    main(args)