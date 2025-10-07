from .utils import pixel_distance

class EventManager:
    def __init__(self, cfg):
        """
        Manages the detection of in-game events.

        Args:
            cfg (dict): The application configuration dictionary.
        """
        self.cfg = cfg

        # State tracking
        self.last_owner_pid = None
        self.current_owner_pid = None
        self.possession_start_frame = 0
        self.possession_start_pos = None

    def update(self, frame_idx, players, ball, new_owner_pid):
        """
        Updates the event manager with the latest frame data and returns any new events.

        Args:
            frame_idx (int): The current frame index.
            players (dict): Dictionary of all player data.
            ball (dict): Dictionary with ball data.
            new_owner_pid (int or None): The ID of the player currently owning the ball.

        Returns:
            list: A list of event dictionaries detected in this frame.
        """
        events = []
        self.last_owner_pid = self.current_owner_pid
        self.current_owner_pid = new_owner_pid

        # --- Pass Detection ---
        if self.last_owner_pid and self.current_owner_pid and self.last_owner_pid != self.current_owner_pid:
            p_from = players.get(self.last_owner_pid)
            p_to = players.get(self.current_owner_pid)
            if p_from and p_to:
                events.append({
                    'type': 'pass',
                    'frame': frame_idx,
                    'from_player_id': self.last_owner_pid,
                    'to_player_id': self.current_owner_pid,
                    'team_id': p_from.get('team')
                })
                # Reset dribble state after a pass
                self._reset_possession_state(frame_idx, p_to)

        # --- Dribble Detection ---
        # If the owner is the same, check for movement
        elif self.last_owner_pid and self.current_owner_pid and self.last_owner_pid == self.current_owner_pid:
            player_data = players.get(self.current_owner_pid)
            if player_data and self.possession_start_pos:
                dist_moved = pixel_distance(self.possession_start_pos, player_data['last_pos'])

                # Convert pixel distance to meters
                dist_moved_m = dist_moved * self.cfg.get('pixels_to_meters', 0.1)

                if dist_moved_m > self.cfg.get('dribble_distance_threshold_m', 5):
                    events.append({
                        'type': 'dribble',
                        'frame': frame_idx,
                        'player_id': self.current_owner_pid,
                        'team_id': player_data.get('team'),
                        'distance_m': round(dist_moved_m, 2)
                    })
                    # Reset after detecting a dribble to start a new one
                    self._reset_possession_state(frame_idx, player_data)

        # If there's a new owner, reset possession state
        elif self.current_owner_pid and self.last_owner_pid != self.current_owner_pid:
            self._reset_possession_state(frame_idx, players.get(self.current_owner_pid))

        # --- Shot Detection (simplified) ---
        # A shot is a fast-moving ball after a player interaction.
        # This requires ball speed, which we'll need to calculate in the main loop.
        # For now, we can add a placeholder.

        return events

    def _reset_possession_state(self, frame_idx, player_data):
        """Resets the state for tracking possession for dribbling."""
        self.possession_start_frame = frame_idx
        if player_data:
            self.possession_start_pos = player_data.get('last_pos')
        else:
            self.possession_start_pos = None