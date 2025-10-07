import cv2
import numpy as np

# --- Configuration des couleurs ---
# Utiliser un colormap pour assigner des couleurs uniques et vives aux joueurs
PLAYER_COLORS = cv2.applyColorMap(np.arange(0, 255, 15, dtype=np.uint8), cv2.COLORMAP_HSV)
BALL_COLOR = (255, 255, 255)  # Blanc
TEAM_A_COLOR = (255, 0, 0)   # Bleu
TEAM_B_COLOR = (0, 0, 255)   # Rouge

def draw_annotations(frame, players, ball_position, team_assignments):
    """
    Dessine toutes les annotations sur une image du match.

    Args:
        frame (np.array): L'image vidéo sur laquelle dessiner.
        players (dict): Dictionnaire contenant les informations des joueurs (positions, etc.).
        ball_position (tuple): Coordonnées (x, y) du ballon.
        team_assignments (dict): Dictionnaire associant les ID de joueurs à leur équipe ('A' ou 'B').

    Returns:
        np.array: L'image avec les annotations.
    """
    # --- Dessiner les trajectoires ---
    for pid, player_data in players.items():
        positions = np.array(player_data.get('positions', []), dtype=np.int32).reshape((-1, 1, 2))
        if len(positions) > 2:
            color = PLAYER_COLORS[pid % len(PLAYER_COLORS)][0].tolist()
            cv2.polylines(frame, [positions], isClosed=False, color=color, thickness=2)

    # --- Dessiner les boîtes des joueurs et leur ID ---
    for pid, player_data in players.items():
        if player_data.get('last_pos'):
            x, y = int(player_data['last_pos'][0]), int(player_data['last_pos'][1])
            team = team_assignments.get(pid)

            # Choisir la couleur en fonction de l'équipe
            color = TEAM_A_COLOR if team == '0' else TEAM_B_COLOR if team == '1' else (0, 255, 0)

            # Dessiner un cercle pour représenter le joueur
            cv2.circle(frame, (x, y), radius=10, color=color, thickness=-1)
            cv2.putText(frame, str(pid), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # --- Dessiner le ballon ---
    if ball_position:
        cv2.circle(frame, ball_position, radius=8, color=BALL_COLOR, thickness=-1)

    return frame

def generate_video(input_path, output_path, analysis_data):
    """
    Génère une vidéo annotée à partir des données d'analyse.
    (Cette fonction sera implémentée plus en détail plus tard)
    """
    print(f"Génération de la vidéo annotée vers {output_path}...")
    # Logique pour lire la vidéo, appliquer draw_annotations frame par frame, et sauvegarder.
    pass

def create_heatmap(positions, frame_shape):
    """
    Crée une heatmap à partir d'une liste de positions.
    (Cette fonction sera implémentée plus en détail plus tard)
    """
    print("Génération de la heatmap...")
    # Logique pour créer une heatmap à partir des coordonnées.
    pass