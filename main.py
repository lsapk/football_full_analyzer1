import argparse, os, json, time
import cv2, numpy as np, pandas as pd
from detector import Detector
from tracker import parse_frame_results
from utils import load_config, box_center, pixel_distance, speed_kmh
from events import detect_pass
from ocr_numero import JerseyOCR

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
parser.add_argument('--output', default='results')
parser.add_argument('--config', default='config.example.json')
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
cfg = load_config(args.config)
detector = Detector(model_name=cfg.get('yolo_model','yolov8n.pt'))
ocr = JerseyOCR(langs=['en'])

results_iter = detector.detect(args.video, show=False)

players = {}  # track_id -> stats
ball_tracks = {}
events = []
last_owner = None
last_ball_time = None
fps = 25.0

frame_idx = 0
start = time.time()
print('Starting processing...')
for res in results_iter:
    frame_time = frame_idx / fps
    persons, balls = parse_frame_results(res, detector)
    # pick primary ball
    ball = balls[0] if len(balls)>0 else None
    bx = by = None
    if ball:
        bx,by = box_center(ball['box'])
        bid = ball['id']
        if bid is not None:
            ball_tracks.setdefault(bid, []).append((frame_idx, bx, by))
        if last_ball_time is None:
            last_ball_time = frame_time

    # mark nearest player as owner
    owner = None
    if ball and persons:
        dists = []
        for p in persons:
            cx,cy = box_center(p['box'])
            d = pixel_distance((cx,cy),(bx,by))
            dists.append((d,p))
        dists = sorted(dists, key=lambda x:x[0])
        owner = dists[0][1]
        pid = owner['id']
        if pid is not None:
            players.setdefault(pid, {'touches':0,'positions':[],'dist_pixels':0.0,'last_pos':None,'last_frame':None,'max_speed_kmh':0.0,'number':None})
            players[pid]['touches'] += 1
            cx,cy = box_center(owner['box'])
            players[pid]['positions'].append((frame_idx,cx,cy))
            if players[pid]['last_pos'] is not None:
                dd = pixel_distance(players[pid]['last_pos'],(cx,cy))
                players[pid]['dist_pixels'] += dd
                dt = (frame_idx - players[pid]['last_frame']) / fps if players[pid]['last_frame'] is not None else 1.0/fps
                sp = speed_kmh(dd, dt if dt>0 else 1.0/fps, cfg.get('pixels_to_meters',0.01))
                if sp > players[pid]['max_speed_kmh']:
                    players[pid]['max_speed_kmh'] = sp
            players[pid]['last_pos'] = (cx,cy)
            players[pid]['last_frame'] = frame_idx
            # try OCR to read number occasionally
            if len(players[pid]['positions']) % 50 == 0:
                # attempt to read number from frame image
                try:
                    frame = res.orig_img if hasattr(res,'orig_img') else None
                    if frame is None:
                        # no frame image available, skip OCR
                        pass
                    else:
                        num = ocr.read_number(frame, owner['box'])
                        if num:
                            players[pid]['number'] = num
                except Exception:
                    pass

    # detect pass event
    if owner is not None and last_owner is not None and last_owner != owner['id']:
        tdiff = frame_time - (last_ball_time if last_ball_time is not None else frame_time)
        try:
            prev_pos = players.get(last_owner, {}).get('last_pos', None)
            curr_pos = box_center(owner['box'])
            if prev_pos and detect_pass(last_owner, owner['id'], prev_pos, curr_pos, tdiff, cfg):
                events.append({'type':'pass','time_s':frame_time,'from':last_owner,'to':owner['id']})
        except Exception:
            pass

    last_owner = owner['id'] if owner is not None else last_owner
    frame_idx += 1

# export results
rows = []
for pid, d in players.items():
    rows.append({'player_id': pid, 'number': d.get('number'), 'touches': d.get('touches'), 'distance_m': d.get('dist_pixels',0.0)*cfg.get('pixels_to_meters',0.01), 'max_speed_kmh': d.get('max_speed_kmh',0.0) })

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.output,'players_stats.csv'), index=False)
ev_df = pd.DataFrame(events)
ev_df.to_csv(os.path.join(args.output,'events.csv'), index=False)
summary = {'video':args.video, 'n_players': len(players), 'n_events': len(events)}
import json
with open(os.path.join(args.output,'summary.json'),'w') as f:
    json.dump(summary,f, indent=2)
print('Done. Results saved in', args.output)
