import math, json, os
import numpy as np
from shapely.geometry import Point, Polygon
import cv2

def get_player_team(frame, player_box, team_a_colors, team_b_colors):
    # crop player image
    x1,y1,x2,y2 = [int(v) for v in player_box]
    player_img = frame[y1:y2, x1:x2]
    # hsv is better for color detection
    hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

    # sum of pixels matching team colors
    score_a = 0
    for c in team_a_colors:
        lower = np.array(c['lower'])
        upper = np.array(c['upper'])
        mask = cv2.inRange(hsv, lower, upper)
        score_a += cv2.countNonZero(mask)

    score_b = 0
    for c in team_b_colors:
        lower = np.array(c['lower'])
        upper = np.array(c['upper'])
        mask = cv2.inRange(hsv, lower, upper)
        score_b += cv2.countNonZero(mask)

    if score_a > score_b:
        return 'a'
    elif score_b > score_a:
        return 'b'
    return None

def box_center(box):
    # box: [x1,y1,x2,y2]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    return (x,y)

def pixel_distance(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def speed_kmh(pixels, dt_seconds, pixels_to_meters):
    if dt_seconds <= 0: return 0.0
    meters = pixels * pixels_to_meters
    m_per_s = meters / dt_seconds
    return m_per_s * 3.6

def load_config(path='config.example.json'):
    if not os.path.exists(path):
        path = 'config.example.json'
    with open(path,'r') as f:
        return json.load(f)
