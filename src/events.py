from .utils import pixel_distance, box_center

class EventManager:
    def __init__(self, cfg):
        """
        Manages the detection of in-game events based on state changes
        provided by the main analysis loop.

        Args:
            cfg (dict): The application configuration dictionary.
        """
        self.cfg = cfg
        # Assuming frame width is available in config, needed for shot detection
        self.frame_width = cfg.get('frame_width', 1920)

    def update(self, frame_idx, players, ball, last_owner_pid, current_owner_pid):
        """
        Analyzes the change in ball ownership to detect events like passes and shots.

        Args:
            frame_idx (int): The current frame index.
            players (dict): Dictionary of all player data.
            ball (dict): Dictionary with ball data.
            last_owner_pid (int or None): The ID of the player who owned the ball previously.
            current_owner_pid (int or None): The ID of the player who owns the ball now.

        Returns:
            list: A list of event dictionaries detected in this frame.
        """
        events = []
        p_from = players.get(last_owner_pid)

        # Event detection requires a change in ball possession
        if not p_from or not last_owner_pid or last_owner_pid == current_owner_pid:
            return events

        # --- Shot Detection ---
        # A shot is detected if a player loses possession and the ball is in the attacking third.
        # This is a simple heuristic and can be improved.
        if current_owner_pid is None and ball:
            ball_pos = box_center(ball['box'])
            # Assuming team 0 attacks towards the right, team 1 towards the left
            team_attacking_direction = 1 if p_from.get('team') == 0 else -1
            attacking_third_threshold = self.frame_width * (4/5) if team_attacking_direction == 1 else self.frame_width * (1/5)

            is_in_attacking_third = (ball_pos[0] > attacking_third_threshold) if team_attacking_direction == 1 else (ball_pos[0] < attacking_third_threshold)

            if is_in_attacking_third:
                events.append({
                    'type': 'shot',
                    'frame': frame_idx,
                    'player_id': last_owner_pid,
                    'team_id': p_from.get('team'),
                    'start_pos': p_from.get('last_pos'),
                    'end_pos': ball_pos
                })
                return events # Prioritize shot over other events if conditions are met

        # --- Pass Detection ---
        # A pass occurs when the ball ownership changes from one player to another on the same team.
        if current_owner_pid:
            p_to = players.get(current_owner_pid)
            if p_to and p_from.get('team') is not None and p_from.get('team') == p_to.get('team'):
                start_pos = p_from.get('last_pos')
                end_pos = p_to.get('last_pos')
                pass_length = pixel_distance(start_pos, end_pos) * self.cfg.get('pixels_to_meters', 0.1) if start_pos and end_pos else 0

                events.append({
                    'type': 'pass',
                    'frame': frame_idx,
                    'from_player_id': last_owner_pid,
                    'to_player_id': current_owner_pid,
                    'team_id': p_from.get('team'),
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'length_m': round(pass_length, 2)
                })

        return events