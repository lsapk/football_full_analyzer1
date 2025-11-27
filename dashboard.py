import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import os
import tempfile
import base64
from src.main import run_analysis
from src.visualization import generate_annotated_video
import diskcache

# --- Configuration & Initialisation ---
UPLOAD_DIRECTORY = tempfile.gettempdir()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

DEFAULT_CONFIG = {
    "inactive_game_frame_limit": 50, "min_player_positions": 15,
    "pixels_to_meters": 0.1, "team_clustering_sample_frames": 20,
    "team_names": {"0": "Équipe A", "1": "Équipe B"}, "ocr_interval": 25
}

# --- Fonctions Utilitaires ---
def load_data_from_db(db_path, game_id=1):
    if not os.path.exists(db_path): return pd.DataFrame(), pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        events_df = pd.read_sql_query(f"SELECT * FROM events WHERE game_id = {game_id}", conn)
        players_df = pd.read_sql_query(f"SELECT * FROM players WHERE game_id = {game_id}", conn)
        conn.close()
        return events_df, players_df
    except Exception: return pd.DataFrame(), pd.DataFrame()

def create_field_figure():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, line_width=0, fillcolor="#53a25b", layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color="white", width=2), layer="below")
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="white", width=2), layer="below")
    fig.add_shape(type="circle", x0=52.5-9.15, y0=34-9.15, x1=52.5+9.15, y1=34+9.15, line=dict(color="white", width=2), layer="below")
    fig.add_shape(type="rect", x0=0, y0=13.85, x1=16.5, y1=54.15, line=dict(color="white", width=2), layer="below")
    fig.add_shape(type="rect", x0=105-16.5, y0=13.85, x1=105, y1=54.15, line=dict(color="white", width=2), layer="below")
    fig.update_layout(xaxis=dict(range=[0, 105], showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(range=[0, 68], showgrid=False, zeroline=False, showticklabels=False),
                      showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=500,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Layout de l'application ---
app.layout = dbc.Container([
    dcc.Store(id='analysis-results-store'),
    dcc.Store(id='video-info-store'),

    dbc.Row(dbc.Col(html.H1("⚽ AI Football Analyst", className="text-center text-white my-4"))),

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # Colonne de gauche: Upload et Contrôles
                dbc.Col([
                    html.H4("1. Charger une vidéo", className="card-title"),
                    dcc.Upload(id='upload-video',
                               children=html.Div(['Glissez-déposez ou ', html.A('sélectionnez une vidéo')]),
                               style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                                      'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '10px'}),
                    html.Div(id='upload-status', className="text-center"),

                    html.H4("2. Choisir la qualité", className="card-title mt-4"),
                    dcc.RadioItems(id='quality-selector', className="text-white",
                                   options={"15": "⚡ Rapide", "8": "👌 Équilibrée", "2": "🔬 Détaillée"},
                                   value="8", inline=True),

                    html.H4("3. Lancer l'analyse", className="card-title mt-4"),
                    dbc.Button("Analyser", id='run-analysis-button', n_clicks=0, className="w-100", disabled=True)
                ], width=4),

                # Colonne de droite: Prévisualisation
                dbc.Col([
                    html.Div(id='video-preview-container', style={'display': 'none'}, children=[
                        html.H4("Prévisualisation", className="card-title"),
                        html.Video(id='video-preview', controls=True, style={'width': '100%'})
                    ])
                ], width=8)
            ]),
        ]),
        className="mb-4",
    ),

    # Section de Progression et Résultats
    dbc.Card(
        dbc.CardBody([
            # Progression
            dbc.Row(id='progress-section', style={'display': 'none'}, children=[
                dbc.Col([
                    html.H4("Analyse en cours..."),
                    dbc.Progress(id='progress-bar', value=0, style={"height": "20px"}),
                    html.Pre(id='progress-log', className="text-white small", style={'height': '150px', 'overflowY': 'scroll', 'backgroundColor': '#1E2938', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'})
                ])
            ]),
            # Résultats
            dbc.Row(id='results-section', style={'display': 'none'}, children=[
                # Colonne de gauche: Vidéo Annotée
                dbc.Col([
                    html.H4("Vidéo Annotée"),
                    html.Div(id='annotated-video-container', children="Génération de la vidéo..."),
                    dbc.Button("Télécharger la vidéo", id="download-video-button", className="mt-2 w-100", disabled=True),
                    dcc.Download(id="download-video")
                ], width=6),

                # Colonne de droite: Visualisation Tactique
                dbc.Col([
                    dcc.Graph(id='football-pitch', figure=create_field_figure()),
                    html.Div(id='events-table-container')
                ], width=6)
            ])
        ])
    ),
], fluid=True)


# --- Callbacks ---

@app.callback(
    [Output('video-info-store', 'data'),
     Output('upload-status', 'children'),
     Output('run-analysis-button', 'disabled'),
     Output('video-preview-container', 'style'),
     Output('video-preview', 'src')],
    [Input('upload-video', 'contents')],
    [State('upload-video', 'filename')]
)
def handle_upload(contents, filename):
    if contents:
        content_type, content_string = contents.split(',')
        video_path = os.path.join(UPLOAD_DIRECTORY, filename)
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(content_string))

        video_info = {'path': video_path, 'name': filename}
        status_msg = f"Fichier '{filename}' chargé."
        return video_info, status_msg, False, {'display': 'block'}, contents
    return {}, "", True, {'display': 'none'}, ''

@app.callback(
    [Output('analysis-results-store', 'data'),
     Output('progress-section', 'style'),
     Output('results-section', 'style')],
    [Input('run-analysis-button', 'n_clicks')],
    [State('video-info-store', 'data'), State('quality-selector', 'value')],
    background=True,
    progress=[Output('progress-bar', 'value'), Output('progress-log', 'children')],
    prevent_initial_call=True
)
def run_analysis_callback(set_progress, n_clicks, video_info, frame_skip):
    if not n_clicks or not video_info: return dash.no_update

    video_path = video_info['path']
    video_name = os.path.splitext(video_info['name'])[0]
    output_dir = os.path.join(UPLOAD_DIRECTORY, f"output_{video_name}")
    os.makedirs(output_dir, exist_ok=True)

    log_messages = []
    def progress_callback(message):
        log_messages.append(message)
        log_output = "\n".join(log_messages[-15:])
        percent_complete = 0
        if "Frame" in message and "/" in message:
            try:
                frame_part = message.split(" ")[1]
                current, total = map(int, frame_part.split('/'))
                percent_complete = int((current / total) * 95)
            except Exception: pass
        elif "Phase 3/3" in message: percent_complete = 98
        set_progress((percent_complete, log_output))

    current_config = DEFAULT_CONFIG.copy()
    current_config['frame_skip'] = int(frame_skip)

    results = run_analysis(video_path=video_path, output_dir=output_dir, model_path='models/yolov8l.pt', config=current_config, progress_callback=progress_callback)

    set_progress((100, "Analyse terminée !"))
    return results, {'display': 'none'}, {'display': 'block'}

@app.callback(
    [Output('annotated-video-container', 'children'),
     Output('download-video-button', 'disabled')],
    [Input('analysis-results-store', 'data')],
    prevent_initial_call=True
)
def generate_and_display_video(results):
    if not results: return "Erreur: Aucune donnée d'analyse trouvée.", True

    video_name = os.path.splitext(os.path.basename(results['video_path']))[0]
    output_dir = os.path.dirname(results['annotations_data'])
    annotated_video_path = os.path.join(output_dir, f"{video_name}_annotated.avi")

    if not os.path.exists(annotated_video_path):
        generate_annotated_video(results['video_path'], annotated_video_path, results['annotations_data'])

    return "Vidéo annotée prête pour le téléchargement.", False


@app.callback(
    Output("download-video", "data"),
    Input("download-video-button", "n_clicks"),
    State('analysis-results-store', 'data'),
    prevent_initial_call=True,
)
def download_video_callback(n_clicks, results):
    if not n_clicks or not results: return dash.no_update

    video_name = os.path.splitext(os.path.basename(results['video_path']))[0]
    output_dir = os.path.dirname(results['annotations_data'])
    annotated_video_path = os.path.join(output_dir, f"{video_name}_annotated.avi")

    return dcc.send_file(annotated_video_path)


@app.callback(
    Output('events-table-container', 'children'),
    Input('analysis-results-store', 'data')
)
def update_events_table(results):
    if not results: return "Chargez une vidéo et lancez l'analyse pour voir les événements."

    events_df, _ = load_data_from_db(results['db_path'])
    if events_df.empty: return "Aucun événement détecté."

    return dash_table.DataTable(
        id='events-table',
        columns=[{"name": i, "id": i} for i in events_df.columns],
        data=events_df.to_dict('records'),
        row_selectable='single', page_size=8,
        style_table={'overflowX': 'auto'},
        style_cell={'backgroundColor': '#2b3e50', 'color': 'white', 'textAlign': 'left'},
        style_header={'backgroundColor': '#1E2938', 'fontWeight': 'bold'}
    )

@app.callback(
    Output('football-pitch', 'figure'),
    [Input('events-table', 'selected_rows')],
    [State('analysis-results-store', 'data')],
    prevent_initial_call=True
)
def update_pitch_on_selection(selected_rows, results):
    fig = create_field_figure()
    if not selected_rows or not results: return fig

    events_df, _ = load_data_from_db(results['db_path'])
    selected_event = events_df.iloc[selected_rows[0]]

    fig.add_trace(go.Scatter(x=[selected_event['start_x'], selected_event['end_x']],
                             y=[selected_event['start_y'], selected_event['end_y']],
                             mode='lines+markers', line=dict(color='yellow', width=2),
                             marker=dict(color='yellow', size=8, symbol='circle-open')))
    fig.add_trace(go.Scatter(x=[selected_event['start_x']], y=[selected_event['start_y']],
                             mode='markers', marker=dict(color='cyan', size=12, symbol='x'), name='Start'))

    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
