import sqlite3
import pandas as pd

def create_connection(db_file):
    """Crée une connexion à la base de données SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connexion à SQLite DB version {sqlite3.version} réussie.")
    except sqlite3.Error as e:
        print(e)
    return conn

def create_tables(conn):
    """Crée les tables de la base de données si elles n'existent pas."""
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                player_id INTEGER NOT NULL,
                team_name TEXT,
                touches INTEGER,
                passes INTEGER,
                shots INTEGER,
                distance_m REAL,
                max_speed_kmh REAL,
                FOREIGN KEY (game_id) REFERENCES games (id)
            );
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                team_name TEXT NOT NULL,
                possession_pct REAL,
                total_passes INTEGER,
                total_shots INTEGER,
                avg_compactness REAL,
                FOREIGN KEY (game_id) REFERENCES games (id)
            );
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER,
                frame INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                team_name TEXT,
                player_id INTEGER,
                start_x REAL,
                start_y REAL,
                end_x REAL,
                end_y REAL,
                FOREIGN KEY (game_id) REFERENCES games (id)
            );
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"Erreur lors de la création des tables : {e}")

def save_analysis_to_db(db_path, video_name, players_df, teams_df, events_df):
    """Enregistre les résultats complets d'une analyse dans la base de données."""
    conn = create_connection(db_path)
    if conn is None:
        return

    create_tables(conn)
    cursor = conn.cursor()

    # 1. Insérer le match et obtenir son ID
    cursor.execute("INSERT INTO games (video_name) VALUES (?)", (video_name,))
    game_id = cursor.lastrowid

    # 2. Préparer et insérer les données des joueurs
    if not players_df.empty:
        players_df['game_id'] = game_id
        players_df.to_sql('players', conn, if_exists='append', index=False)

    # 3. Préparer et insérer les données des équipes
    if not teams_df.empty:
        teams_df['game_id'] = game_id
        teams_df.to_sql('teams', conn, if_exists='append', index=False)

    # 4. Préparer et insérer les données des événements
    if not events_df.empty:
        events_df['game_id'] = game_id
        events_df.to_sql('events', conn, if_exists='append', index=False)

    conn.close()
    print(f"Analyse pour '{video_name}' sauvegardée avec succès dans {db_path}")
