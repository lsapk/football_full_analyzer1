from .utils import pixel_distance

class EventManager:
    def __init__(self, cfg):
        """
        Manages the detection of in-game events based on state changes
        provided by the main analysis loop.

        Args:
            cfg (dict): The application configuration dictionary.
        """
        self.cfg = cfg

    def update(self, frame_idx, players, ball, last_owner_pid, current_owner_pid):
        """
        Analyzes the change in ball ownership to detect events like passes.

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

        # --- Pass Detection ---
        # A pass occurs when the ball ownership changes from one player to another on the same team.
        if last_owner_pid and current_owner_pid and last_owner_pid != current_owner_pid:
            p_from = players.get(last_owner_pid)
            p_to = players.get(current_owner_pid)

            # Ensure both players exist and are on the same team for a pass
            if p_from and p_to and p_from.get('team') is not None and p_from.get('team') == p_to.get('team'):
                events.append({
                    'type': 'pass',
                    'frame': frame_idx,
                    'from_player_id': last_owner_pid,
                    'to_player_id': current_owner_pid,
                    'team_id': p_from.get('team')
                })

        return events