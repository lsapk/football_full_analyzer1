# ⚽ AI Football Analyst

Cette application utilise l'intelligence artificielle pour analyser des courtes vidéos de matchs de football (≤ 5 minutes). Elle détecte et suit les joueurs et le ballon, extrait des statistiques de performance, identifie des événements de jeu clés et génère un rapport tactique automatisé à l'aide d'un Grand Modèle de Langage (LLM).

## 🚀 Fonctionnalités

- **🎥 Analyse Vidéo Automatisée** : Traite un fichier vidéo pour identifier les joueurs et le ballon.
- **👥 Identification d'Équipe par Clustering** : Assigne automatiquement les joueurs à deux équipes sans configuration manuelle des couleurs.
- **📊 Statistiques Complètes** : Calcule des statistiques par joueur (distance, vitesse, touches) et par équipe (possession, compacité, nombre de passes/dribbles).
- **📹 Vidéo Annotée** : Génère une vidéo de sortie avec les joueurs, leurs trajectoires et la compacité de l'équipe affichés en temps réel.
- **🧠 Analyse Tactique par IA (Optionnel)** : Utilise un LLM (GPT) pour générer un rapport texte analysant la stratégie des équipes, leurs forces, faiblesses et des suggestions d'amélioration.
- **💾 Export de Données** : Sauvegarde toutes les statistiques et les événements dans des fichiers CSV pour une analyse plus approfondie.

## 🛠️ Technologies Principales

- **Python**
- **YOLOv8 & BoT-SORT** : Pour la détection et le suivi des objets.
- **OpenCV** : Pour le traitement vidéo.
- **Pandas & NumPy** : Pour la manipulation et l'analyse des données.
- **Scikit-learn** : Pour le clustering des équipes.
- **OpenAI API** : Pour la génération du rapport tactique.

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

4.  **(Optionnel) Pour l'analyse tactique par IA :**
    Vous devez avoir une clé API d'OpenAI. Définissez-la comme une variable d'environnement :
    ```bash
    export OPENAI_API_KEY="votre_cle_api_ici"
    # Sur Windows (cmd): set OPENAI_API_KEY="votre_cle_api_ici"
    ```

## ▶️ Utilisation

Lancez l'analyse avec la commande suivante, en spécifiant le chemin vers votre vidéo.

**Commande de base :**
```bash
python main.py --video data/votre_video.mp4 --output results
```

**Avec l'analyse tactique par IA :**
```bash
python main.py --video data/votre_video.mp4 --output results --llm
```

### Arguments

- `--video` : (Requis) Chemin vers le fichier vidéo à analyser.
- `--output` : (Optionnel) Dossier où sauvegarder les résultats. Par défaut : `output/`.
- `--model` : (Optionnel) Chemin vers le modèle YOLOv8. Par défaut : `models/yolov8n.pt`.
- `--llm` : (Optionnel) Active la génération du rapport tactique par le LLM.

## 📁 Fichiers de Sortie

Après une analyse réussie, vous trouverez les fichiers suivants dans votre dossier de sortie :

- `*_annotated.mp4` : La vidéo originale, annotée avec les boîtes des joueurs, leurs trajectoires et la compacité de l'équipe.
- `players_stats.csv` : Statistiques détaillées pour chaque joueur (distance, vitesse, etc.).
- `team_stats.csv` : Statistiques agrégées pour chaque équipe (possession, compacité, passes, dribbles).
- `events.csv` : Liste de tous les événements détectés (passes, dribbles) avec les détails.
- `summary.json` : Un résumé simple de l'analyse.
- `tactical_report.txt` : (Si `--llm` est utilisé) Le rapport d'analyse généré par l'IA.