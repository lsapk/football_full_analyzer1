import streamlit as st
import os
import tempfile
import json
import cv2
import numpy as np
import pandas as pd
from src.main import run_analysis
from src.visualization import draw_annotations

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "frame_skip": 2,
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

st.set_page_config(page_title="Analyseur de Match de Football", layout="wide")

st.title("⚽ Analyseur de Match de Football Interactif")

# --- Session State Initialization ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# --- Main App Logic ---
uploaded_file = st.file_uploader("Choisissez une vidéo...", type=["mp4", "mov", 'avi'])

if uploaded_file:
    # Use a persistent path for the video file
    temp_dir = tempfile.gettempdir()
    st.session_state.video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(st.session_state.video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(st.session_state.video_path)

    # --- Analysis Quality Selector ---
    quality_options = {"⚡ Rapide (1 image / 15)": 15, "👌 Équilibrée (1 image / 8)": 8, "🔬 Détaillée (1 image / 2)": 2}
    selected_quality = st.radio("Qualité de l'analyse:", options=list(quality_options.keys()), index=1, horizontal=True)
    frame_skip = quality_options[selected_quality]

    if st.button("Lancer l'analyse"):
        st.session_state.analysis_done = False
        st.session_state.results = None

        # Create a persistent output directory based on video name
        video_name = os.path.splitext(uploaded_file.name)[0]
        output_dir = os.path.join(temp_dir, f"output_{video_name}")
        os.makedirs(output_dir, exist_ok=True)

        log_container = st.container()
        log_container.header("📈 Progression de l'analyse")
        progress_bar = log_container.progress(0, "Démarrage...")
        log_area = log_container.empty()
        log_messages = []

        current_config = DEFAULT_CONFIG.copy()
        current_config['frame_skip'] = frame_skip

        def progress_callback(message):
            """Callback pour mettre à jour l'interface Streamlit."""
            log_messages.append(message)
            # Affiche les 15 derniers messages
            log_output = "\n".join(log_messages[-15:])
            log_area.code(log_output, language='bash')

            if "Phase 1/2" in message:
                 progress_bar.progress(0, "Phase 1/2 : Analyse vidéo...")
            elif "Frame" in message and "/" in message:
                try:
                    # Extrait la progression depuis le message de log
                    frame_part = message.split(" ")[1]
                    current, total = map(int, frame_part.split('/'))
                    percent_complete = int((current / total) * 100)
                    progress_bar.progress(percent_complete, f"Phase 1/2 : Analyse Vidéo ({percent_complete}%)")
                except (ValueError, IndexError):
                    pass # Ignore les messages de log qui ne correspondent pas
            elif "Phase 2/2" in message:
                progress_bar.progress(95, "Phase 2/2 : Finalisation...")

        results = run_analysis(
            video_path=st.session_state.video_path,
            output_dir=output_dir,
            model_path='models/yolov8l.pt', # Using a larger, more accurate model
            config=current_config,
            progress_callback=progress_callback
        )

        progress_bar.progress(100, "Analyse terminée !")
        st.session_state.results = results
        st.session_state.analysis_done = True
        st.rerun()

# --- Video Player and Statistics ---
if st.session_state.analysis_done and st.session_state.results:
    st.success("Analyse terminée ! La vidéo annotée est prête.")

    # Define the output path for the annotated video
    video_name = os.path.splitext(os.path.basename(st.session_state.video_path))[0]
    output_dir = os.path.dirname(st.session_state.results['annotations_data'])
    annotated_video_path = os.path.join(output_dir, f"{video_name}_annotated.avi")

    # Check if video already exists
    if not os.path.exists(annotated_video_path):
        with st.spinner("Génération de la vidéo annotée... (cela peut prendre quelques minutes)"):
            # Generate the annotated video
            from src.visualization import generate_annotated_video
            generate_annotated_video(
                st.session_state.video_path,
                annotated_video_path,
                st.session_state.results['annotations_data']
            )

    st.video(annotated_video_path)

    with open(annotated_video_path, "rb") as file:
        st.download_button(
            label="Télécharger la vidéo annotée",
            data=file,
            file_name=os.path.basename(annotated_video_path),
            mime="video/avi"
        )

    st.subheader("Statistiques de l'Analyse")

    st.info(f"Les données de l'analyse ont été sauvegardées dans la base de données : {st.session_state.results['db_path']}")