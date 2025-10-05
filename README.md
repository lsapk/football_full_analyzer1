# Football Full Analyzer - Version complète (Prototype)
Ce projet est une application locale complète pour analyser des vidéos de match (>90 min).
Il inclut la détection et le suivi des joueurs (YOLOv8), l'identification des équipes par couleur de maillot, et le calcul de statistiques par joueur et par équipe.

## Installation
1.  Décompressez l'archive et ouvrez un terminal dans le dossier.
2.  (Recommandé) Créez et activez un environnement virtuel :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```
3.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```
4.  Téléchargez les modèles d'IA nécessaires :
    ```bash
    python download_models.py
    ```

## Lancement de l'analyse
Pour lancer l'analyse, utilisez la commande suivante dans votre terminal, en remplaçant `/chemin/vers/votre_video.mp4` par le chemin réel de votre fichier vidéo :
```bash
python main.py --video /chemin/vers/votre_video.mp4 --output results
```
Les résultats seront sauvegardés dans le dossier `results`.

---

## Optimisation des performances
L'analyse vidéo est une tâche très gourmande en ressources. Si vous trouvez que l'analyse est trop lente, voici deux méthodes très efficaces pour l'accélérer.

### Méthode 1 : Alléger la vidéo en amont (Recommandé)
La méthode la plus efficace est de réduire la qualité de la vidéo avant de la passer à l'analyseur. Une résolution plus faible et moins d'images par seconde (fps) n'empêchent pas l'IA de faire son travail correctement.

Vous pouvez utiliser l'outil gratuit `ffmpeg` pour créer une version "légère" de votre vidéo.

**Exemple de commande à exécuter une seule fois avant l'analyse :**
```bash
ffmpeg -i "votre_video_originale.mp4" -vf "scale=1280:-1,fps=15" -c:v libx264 -preset veryfast -crf 28 -an "video_optimisee.mp4"
```
*   `scale=1280:-1` : redimensionne la vidéo à une largeur de 1280 pixels (720p).
*   `fps=15` : réduit le nombre d'images par seconde à 15.
*   `-crf 28` : compresse la vidéo pour réduire son poids (une valeur plus élevée compresse plus).
*   `-an` : supprime la piste audio pour alléger le fichier.

Lancez ensuite l'analyse sur le fichier `video_optimisee.mp4`. **Cela peut diviser le temps d'analyse par 5 ou 10.**

### Méthode 2 : Ajuster l'échantillonnage dans l'application
Vous pouvez contrôler la précision de l'analyse directement depuis l'application en modifiant le fichier `config.json`.

Ouvrez le fichier `config.json` et modifiez la valeur du paramètre `frame_skip`.
```json
{
  "frame_skip": 25,
  ...
}
```
*   `"frame_skip": 1` : analyse **toutes** les images. Très lent, mais très précis.
*   `"frame_skip": 10` : analyse 1 image sur 10. Bon compromis.
*   `"frame_skip": 25` : analyse 1 image sur 25 (environ 1 image par seconde). **Réglage par défaut, rapide et souvent suffisant.**

Plus la valeur est élevée, plus l'analyse est rapide, mais moins les statistiques seront précises. N'hésitez pas à expérimenter pour trouver le réglage qui vous convient le mieux.

## Configuration avancée
Dans le fichier `config.json`, vous pouvez également :
*   **Changer les couleurs des équipes :** Modifiez les valeurs `lower` et `upper` dans `team_a_colors` et `team_b_colors`. Ce sont des plages de couleurs au format HSV. Des outils en ligne comme "HSV color picker" peuvent vous aider à trouver les bonnes valeurs pour vos maillots.
*   **Changer le modèle d'IA :** Vous pouvez utiliser un modèle YOLO plus gros (ex: `yolov8m.pt`) pour plus de précision, mais au détriment de la vitesse.