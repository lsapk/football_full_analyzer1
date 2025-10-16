import streamlit as st
import os
import tempfile
import json
import cv2
import numpy as np
import pandas as pd
from src.main import run_analysis
from main import DEFAULT_CONFIG
from src.visualization import draw_annotations

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
        progress_bar = log_container.progress(0)
        log_area = log_container.empty()
        log_messages = []

        current_config = DEFAULT_CONFIG.copy()
        current_config['frame_skip'] = frame_skip

        analysis_generator = run_analysis(
            video_path=st.session_state.video_path,
            output_dir=output_dir,
            model_path='models/yolov8n.pt',
            config=current_config
        )

        final_results = None
        for message in analysis_generator:
            if isinstance(message, dict):
                final_results = message
                progress_bar.progress(100)
                break

            log_messages.append(str(message))
            log_area.text_area("Logs:", "\n".join(log_messages), height=250)

            if "Frame" in str(message) and "/" in str(message):
                try:
                    frame_part = str(message).split(" ")[1]
                    current, total = map(int, frame_part.split('/'))
                    progress_percent = int((current / total) * 100)
                    progress_bar.progress(progress_percent)
                except (ValueError, IndexError):
                    pass

        st.session_state.results = final_results
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