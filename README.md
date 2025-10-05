# Football Full Analyzer - Version complète (Prototype)
Ce projet est une application locale complète pour analyser des vidéos de match (>90 min).
Il inclut détection+tracking (YOLOv8), OCR pour numéros, calibration du terrain, heuristiques
d'extraction d'événements, une petite interface Flask pour upload/monitor et génération d'outputs.

## Contenu du ZIP
- code Python : main.py, detector.py, tracker.py, events.py, ocr_numero.py, utils.py
- webapp/ : Flask minimal (templates, static)
- scripts : download_models.py, run_full_pipeline.sh
- config.example.json, requirements.txt, README.md

## Installation rapide (Linux/Mac - local)
1. Dézippe l'archive et ouvre un terminal dans le dossier.
2. Crée et active un virtualenv :
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Installe les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Télécharge les modèles nécessaires :
   ```bash
   python download_models.py
   ```
5. (Optionnel) Lance la web UI :
   ```bash
   cd webapp
   flask run --host=0.0.0.0 --port=5000
   ```
   Page d'upload : http://localhost:5000
6. Ou lance directement l'analyse en CLI :
   ```bash
   python main.py --video /chemin/ton_video.mp4 --output results
   ```

## Remarques
- Les poids YOLOv8 sont téléchargés automatiquement par `download_models.py` (yolov8n by default).
- Le pipeline supporte reprise (checkpointing) entre chunks.
- Pour de meilleures performances, installe CUDA et utilisez une machine avec GPU NVIDIA.
- Si les numéros sont illisibles, utilise l'UI pour corriger manuellement les affectations player_id->numero.
