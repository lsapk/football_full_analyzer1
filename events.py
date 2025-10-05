# heuristics to detect passes, shots, and goals based on owners and ball trajectory
from utils import pixel_distance

def detect_pass(prev_owner_id, curr_owner_id, prev_pos, curr_pos, dt, cfg):
    # simple heuristics: owner changed and distance reasonable and time short
    if prev_owner_id is None or curr_owner_id is None: return False
    if prev_owner_id == curr_owner_id: return False
    pd = pixel_distance(prev_pos, curr_pos) * cfg.get('pixels_to_meters', 0.01)
    if dt <= cfg.get('pass_time_threshold_s',3.0) and pd <= cfg.get('pass_distance_threshold_m',12.0):
        return True
    return False
