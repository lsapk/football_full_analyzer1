import os
from src.main import run_analysis

# --- Configuration pour le test ---
VIDEO_PATH = "dummy_video.mp4" # Fichier qui n'existe pas
OUTPUT_DIR = "test_output"
MODEL_PATH = "models/yolov8l.pt"
CONFIG = {
    "frame_skip": 15,
    "inactive_game_frame_limit": 50,
    "min_player_positions": 15,
    "pixels_to_meters": 0.1,
    "team_clustering_sample_frames": 20,
    "team_names": {
        "0": "Équipe A",
        "1": "Équipe B"
    },
    "ocr_interval": 25
}

def test_run_analysis():
    """
    Teste l'exécution de la fonction run_analysis pour s'assurer
    qu'elle ne lève pas d'erreurs d'importation ou de configuration.
    """
    print("Début du test de run_analysis...")
    try:
        # On s'attend à une erreur car la vidéo n'existe pas,
        # mais l'appel lui-même ne devrait pas échouer avant.
        run_analysis(
            video_path=VIDEO_PATH,
            output_dir=OUTPUT_DIR,
            model_path=MODEL_PATH,
            config=CONFIG
        )
    except Exception as e:
        # On s'attend à une erreur liée au fichier vidéo, ce qui est normal.
        # Toute autre erreur (ex: ImportError) serait un problème.
        if "No such file or directory" in str(e) or isinstance(e, FileNotFoundError):
             print("Test réussi : L'erreur attendue (fichier vidéo non trouvé) a été interceptée.")
        elif "could not find" in str(e).lower():
             print("Test réussi : L'erreur attendue (fichier vidéo non trouvé) a été interceptée.")
        else:
            print(f"Test échoué avec une erreur inattendue : {e}")
            raise e

if __name__ == "__main__":
    # Créer un faux fichier vidéo pour que l'initialisation de cv2.VideoCapture réussisse
    with open(VIDEO_PATH, "w") as f:
        f.write("")

    test_run_analysis()

    # Nettoyer le faux fichier
    os.remove(VIDEO_PATH)
