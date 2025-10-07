import numpy as np
from .utils import pixel_distance, speed_kmh
from scipy.spatial.distance import pdist, squareform

def calculate_team_compactness(player_positions):
    """
    Calculates the team compactness based on the average distance between players.
    A lower value means the team is more compact.
    """
    if len(player_positions) < 2:
        return 0

    # Calculate the pairwise distance between all players
    dist_matrix = pdist(np.array(player_positions))

    # The compactness is the mean of these distances
    return np.mean(dist_matrix)

def update_player_movement(player_data, player_obj, frame_idx, fps, cfg):
    """
    Updates a single player's position, distance, and speed based on their movement.
    This modifies the player_data dictionary in place.
    """
    cx, cy = player_obj['center']
    player_data['positions'].append((frame_idx, cx, cy))

    frame_skip = cfg.get('frame_skip', 1)

    # Calculate distance and speed if the player was seen in a recent frame
    if player_data.get('last_pos') is not None and player_data.get('last_frame') is not None:
        # Avoid calculating distance for "teleporting" players (re-identified after long time)
        if (frame_idx - player_data['last_frame']) < (frame_skip * 5): # Increased tolerance
            dd = pixel_distance(player_data['last_pos'], (cx, cy))
            player_data['dist_pixels'] += dd

            dt = (frame_idx - player_data['last_frame']) / fps
            if dt > 0:
                sp = speed_kmh(dd, dt, cfg.get('pixels_to_meters', 0.1))
                if sp > player_data['max_speed_kmh']:
                    player_data['max_speed_kmh'] = sp

    player_data['last_pos'] = (cx, cy)
    player_data['last_frame'] = frame_idx

def calculate_team_stats(players, team_assignments, pixels_to_meters=0.1):
    """
    Calculates aggregate statistics for each team for the current frame.
    """
    team_stats = {}

    # Initialize stats structure for each team
    unique_teams = set(team_assignments.values())
    for team_id in unique_teams:
        if team_id is None: continue
        team_stats[team_id] = {
            'player_count': 0,
            'positions': [],
            'compactness': 0
        }

    # Aggregate current positions from players
    for pid, p_data in players.items():
        team_id = p_data.get('team')
        if team_id in team_stats:
            if p_data.get('last_pos'):
                team_stats[team_id]['positions'].append(p_data['last_pos'])
                team_stats[team_id]['player_count'] += 1

    # Calculate compactness for each team
    for team_id, stats in team_stats.items():
        if stats['player_count'] > 1:
            compactness_pixels = calculate_team_compactness(stats['positions'])
            stats['compactness'] = round(compactness_pixels * pixels_to_meters, 2)

    return team_stats