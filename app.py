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
            model_path='models/yolov8n.pt',
            config=current_config,
            progress_callback=progress_callback
        )

        progress_bar.progress(100, "Analyse terminée !")
        st.session_state.results = results
        st.session_state.analysis_done = True
        st.experimental_rerun()

# --- Interactive Player ---
if st.session_state.analysis_done and st.session_state.results:
    st.success("Analyse terminée ! Utilisez les contrôles ci-dessous pour explorer la vidéo.")

    with open(st.session_state.results['annotations_data']) as f:
        analysis_data = json.load(f)

    annotations = analysis_data['annotations_by_frame']

    st.sidebar.header("Contrôles d'affichage")
    show_boxes = st.sidebar.checkbox("Afficher le suivi des joueurs", True)
    show_team_shape = st.sidebar.checkbox("Afficher la forme tactique", True)

    cap = cv2.VideoCapture(st.session_state.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    selected_frame_idx = st.slider("Naviguer dans la vidéo", 0, total_frames - 1, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame_idx)
    ret, frame = cap.read()

    if ret:
        analysis_frame_skip = analysis_data.get('frame_skip', 1)
        annotation_idx = selected_frame_idx // analysis_frame_skip

        if annotation_idx < len(annotations):
            ann_data = annotations[annotation_idx]
            team_assignments = analysis_data['team_assignments']
            team_colors = {str(k): tuple(v) for k, v in analysis_data['team_colors'].items()}

            players_to_draw = ann_data.get('players', {}) if show_boxes else {}
            ball_pos = ann_data.get('ball_pos')
            player_positions_for_shape = {pid: p_data['center'] for pid, p_data in ann_data.get('players', {}).items() if p_data.get('center')} if show_team_shape else {}

            annotated_frame = draw_annotations(frame, players_to_draw, ball_pos, team_assignments, team_colors, player_positions_for_shape)
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

    cap.release()

    st.subheader("Statistiques de l'analyse")
    tabs = st.tabs(["Équipes", "Joueurs", "Événements"])
    with tabs[0]:
        st.dataframe(pd.read_csv(st.session_state.results['team_stats']))
    with tabs[1]:
        st.dataframe(pd.read_csv(st.session_state.results['player_stats']))
    with tabs[2]:
        if os.path.exists(st.session_state.results['events']) and os.path.getsize(st.session_state.results['events']) > 0:
            st.dataframe(pd.read_csv(st.session_state.results['events']))
        else:
            st.info("Aucun événement majeur détecté.")