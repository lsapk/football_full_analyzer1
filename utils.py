import math, json, os
import numpy as np
from shapely.geometry import Point, Polygon

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
