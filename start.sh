#!/bin/bash

# Créer le répertoire pour les modèles s'il n'existe pas
mkdir -p models

# Vérifier si le modèle existe, sinon le télécharger
MODEL_PATH="models/yolov8n.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Le modèle n'a pas été trouvé. Téléchargement en cours..."
    wget -O "$MODEL_PATH" https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
else
    echo "Le modèle existe déjà."
fi

# Lancer l'application Streamlit
# Render définit automatiquement la variable $PORT
streamlit run app.py --server.port $PORT --server.address 0.0.0.0