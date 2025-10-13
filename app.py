import streamlit as st
import os
import tempfile
# import pandas as pd

# We defer heavy imports to prevent memory issues on startup (Render free tier)
# import cv2
# from src.main import run_analysis
# from main import DEFAULT_CONFIG

st.set_page_config(page_title="Analyseur de Match de Football", layout="wide")

st.title("⚽ Analyseur de Match de Football")

st.write(
    "Téléversez une vidéo d'un match de football pour obtenir une analyse détaillée, "
    "incluant les statistiques des joueurs, des équipes et une vidéo annotée."
)

# --- Session State Initialization ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None


uploaded_file = st.file_uploader("Choisissez une vidéo...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    if st.button("Lancer l'analyse"):
        st.session_state.analysis_done = False
        st.session_state.results = None
        st.session_state.video_bytes = None

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # --- UI for progress ---
            st.subheader("📈 Progression de l'analyse")
            log_placeholder = st.empty()
            progress_bar = st.progress(0)
            log_messages = []

            def progress_callback(message):
                # Update logs
                log_messages.append(message)
                log_placeholder.text_area("Logs en temps réel", "\n".join(log_messages), height=300)

                # Update progress bar
                if "Frame" in message and "/" in message:
                    try:
                        frame_part = message.split(" ")[1]
                        current_frame, total_frames = map(int, frame_part.split('/'))
                        # Phase 1 is detection/tracking (approx. 80% of the work)
                        progress_percent = int((current_frame / total_frames) * 80)
                        progress_bar.progress(progress_percent)
                    except (ValueError, IndexError):
                        pass # Ignore parsing errors
                elif "Phase 2/2" in message:
                    progress_bar.progress(85)
                elif "Analyse terminée" in message:
                    progress_bar.progress(100)

            with st.spinner("Analyse en cours... Cette opération peut prendre plusieurs minutes."):
                # --- Deferred imports ---
                from src.main import run_analysis
                from main import DEFAULT_CONFIG

                model_path = 'models/yolov8n.pt'
                if not os.path.exists(model_path):
                    st.error(f"Le modèle YOLO est introuvable : {model_path}")
                    st.stop()

                results = run_analysis(
                    video_path=video_path,
                    output_dir=output_dir,
                    model_path=model_path,
                    config=DEFAULT_CONFIG,
                    generate_llm_report=False,
                    progress_callback=progress_callback
                )

                # Store results in session state
                st.session_state.results = results
                st.session_state.analysis_done = True

                # Pre-load video bytes to display after rerun
                with open(results['annotated_video'], 'rb') as f:
                    st.session_state.video_bytes = f.read()

        # Trigger a rerun to display results cleanly
        st.experimental_rerun()

# --- Display Results after analysis is done ---
if st.session_state.analysis_done and st.session_state.results:
    import pandas as pd
    st.success("✅ Analyse terminée avec succès !")
    st.balloons()

    results = st.session_state.results

    st.subheader("🎬 Vidéo Annotée")
    st.video(st.session_state.video_bytes)
    st.download_button(
        label="Télécharger la vidéo annotée (.avi)",
        data=st.session_state.video_bytes,
        file_name=os.path.basename(results['annotated_video'])
    )

    # --- Display DataFrames and provide download buttons ---
    tabs = st.tabs(["Statistiques des Équipes", "Statistiques des Joueurs", "Journal des Événements"])

    with tabs[0]:
        if os.path.exists(results['team_stats']):
            team_df = pd.read_csv(results['team_stats'])
            st.dataframe(team_df)
            st.download_button(
                label="Télécharger les stats des équipes (.csv)",
                data=team_df.to_csv(index=False).encode('utf-8'),
                file_name='team_stats.csv'
            )
        else:
            st.warning("Aucune statistique d'équipe n'a été générée.")

    with tabs[1]:
        if os.path.exists(results['player_stats']):
            player_df = pd.read_csv(results['player_stats'])
            st.dataframe(player_df)
            st.download_button(
                label="Télécharger les stats des joueurs (.csv)",
                data=player_df.to_csv(index=False).encode('utf-8'),
                file_name='player_stats.csv'
            )
        else:
            st.warning("Aucune statistique de joueur n'a été générée.")

    with tabs[2]:
        if os.path.exists(results['events']) and os.path.getsize(results['events']) > 0:
            events_df = pd.read_csv(results['events'])
            st.dataframe(events_df)
            st.download_button(
                label="Télécharger le journal des événements (.csv)",
                data=events_df.to_csv(index=False).encode('utf-8'),
                file_name='events.csv'
            )
        else:
            st.info("Aucun événement (passe, tir, etc.) n'a été détecté.")