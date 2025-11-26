import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import os

# --- Configuration & Initialisation ---
DATABASE_PATH = 'output/analysis.db' # Ce chemin devra peut-être être ajusté
app = dash.Dash(__name__)

# --- Fonctions de Chargement de Données ---
def load_data_from_db(db_path, game_id=1):
    """Charge les données d'un match spécifique depuis la base de données."""
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        events_query = f"SELECT * FROM events WHERE game_id = {game_id}"
        players_query = f"SELECT * FROM players WHERE game_id = {game_id}"

        events_df = pd.read_sql_query(events_query, conn)
        players_df = pd.read_sql_query(players_query, conn)

        conn.close()
        return events_df, players_df
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- Création du Terrain de Football ---
def create_field_figure():
    """Crée une figure Plotly représentant un terrain de football."""
    fig = go.Figure()

    # Pelouse
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, line_width=0, fillcolor="#53a25b")

    # Lignes extérieures
    fig.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color="white", width=2))

    # Ligne médiane
    fig.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="white", width=2))

    # Rond central
    fig.add_shape(type="circle", x0=52.5-9.15, y0=34-9.15, x1=52.5+9.15, y1=34+9.15, line=dict(color="white", width=2))

    # Surfaces de réparation
    fig.add_shape(type="rect", x0=0, y0=13.85, x1=16.5, y1=54.15, line=dict(color="white", width=2))
    fig.add_shape(type="rect", x0=105-16.5, y0=13.85, x1=105, y1=54.15, line=dict(color="white", width=2))

    fig.update_layout(
        xaxis=dict(range=[0, 105], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 68], showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=500
    )
    return fig

# --- Layout de l'Application ---
app.layout = html.Div([
    html.H1("Tableau de Bord Tactique"),
    dcc.Graph(id='football-pitch', figure=create_field_figure()),
    html.H3("Événements du Match"),
    html.Div(id='events-table-container')
])

# --- Callbacks pour l'Interactivité ---
@app.callback(
    Output('events-table-container', 'children'),
    Input('football-pitch', 'id') # Se déclenche au chargement
)
def display_events_table(pitch_id):
    events_df, _ = load_data_from_db(DATABASE_PATH)
    if events_df.empty:
        return "Aucun événement à afficher."

    return dash.dash_table.DataTable(
        id='events-table',
        columns=[{"name": i, "id": i} for i in events_df.columns],
        data=events_df.to_dict('records'),
        row_selectable='single',
        page_size=10,
    )

@app.callback(
    Output('football-pitch', 'figure'),
    Input('events-table', 'selected_rows'),
    prevent_initial_call=True
)
def update_pitch_on_selection(selected_rows):
    fig = create_field_figure()
    if not selected_rows:
        return fig

    events_df, players_df = load_data_from_db(DATABASE_PATH)
    selected_event = events_df.iloc[selected_rows[0]]

    # Dessiner le joueur impliqué
    if pd.notna(selected_event['player_id']):
        player_id = selected_event['player_id']
        fig.add_trace(go.Scatter(
            x=[selected_event['start_x']],
            y=[selected_event['start_y']],
            mode='markers',
            marker=dict(color='yellow', size=15, symbol='x'),
            name=f"Joueur {player_id}"
        ))

    # Dessiner la passe ou le tir
    fig.add_trace(go.Scatter(
        x=[selected_event['start_x'], selected_event['end_x']],
        y=[selected_event['start_y'], selected_event['end_y']],
        mode='lines+markers',
        line=dict(color='blue', width=2, dash='dot'),
        marker=dict(color='blue', size=8)
    ))

    return fig

if __name__ == '__main__':
    app.run(debug=True)
