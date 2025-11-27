# ⚽ AI Football Analyst

Cette application web utilise l'intelligence artificielle pour analyser des vidéos de matchs de football. Elle détecte et suit les joueurs et le ballon, extrait des statistiques de performance, identifie les passes et les tirs, et offre un tableau de bord tactique interactif pour visualiser les résultats.

## 🚀 Fonctionnalités

- **📊 Dashboard Interactif Unique** : Une seule interface web pour charger une vidéo, lancer l'analyse et explorer les résultats de manière interactive.
- **🎥 Analyse Vidéo Automatisée** : Traite un fichier vidéo pour identifier les joueurs et le ballon en utilisant le modèle de détection `YOLOv8l`.
- **👥 Identification d'Équipe par Clustering** : Assigne automatiquement les joueurs à deux équipes en se basant sur la couleur de leur maillot.
- **📈 Statistiques Complètes** : Calcule des statistiques par joueur (distance, vitesse, touches, passes, tirs) et par équipe (possession, compacité).
- ** DATABASE SQLite** : Toutes les données de l'analyse (joueurs, équipes, événements) sont stockées dans une base de données SQLite pour une interrogation et une analyse faciles.
- **🗺️ Visualisation Tactique 2D** : Un terrain de football interactif affiche la position et le déroulement des passes et des tirs.
- **📹 Vidéo Annotée** : Génère une vidéo de sortie avec les joueurs et leurs mouvements, téléchargeable directement depuis l'interface.

## 🛠️ Technologies Principales

- **Python**
- **Dash & Plotly** : Pour le tableau de bord web interactif.
- **YOLOv8 & ByteTrack** : Pour la détection et le suivi des objets.
- **OpenCV** : Pour le traitement vidéo.
- **Pandas & NumPy** : Pour la manipulation et l'analyse des données.
- **Scikit-learn** : Pour le clustering des équipes.
- **SQLite** : Pour le stockage des données.

## ⚙️ Installation

1.  **Clonez le dépôt :**
    ```bash
    git clone <URL_DU_REPO>
    cd <NOM_DU_DOSSIER>
    ```

2.  **(Recommandé) Créez et activez un environnement virtuel :**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Utilisation

1.  **Lancez l'application web :**
    ```bash
    python dashboard.py
    ```

2.  **Ouvrez votre navigateur :**
    Rendez-vous à l'adresse [http://127.0.0.1:8050](http://127.0.0.1:8050).

3.  **Analysez votre vidéo :**
    - Chargez un fichier vidéo en utilisant la zone de glisser-déposer.
    - Choisissez la qualité de l'analyse (Rapide, Équilibrée, ou Détaillée).
    - Cliquez sur le bouton "Analyser" et suivez la progression en temps réel.
    - Une fois l'analyse terminée, le tableau de bord interactif s'affichera avec les résultats.

## 📁 Fichiers de Sortie

L'analyse génère les fichiers suivants dans un sous-dossier du répertoire temporaire de votre système :

- `analysis.db` : Une base de données SQLite contenant toutes les statistiques et les événements de l'analyse.
- `*_annotated.avi` : La vidéo originale, annotée avec les mouvements des joueurs, téléchargeable depuis l'interface.
