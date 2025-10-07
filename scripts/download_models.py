# Script simple pour forcer le téléchargement des poids YOLOv8 via ultralytics.
from ultralytics import YOLO
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='yolov8n.pt', help='Nom du modèle YOLOv8 à télécharger')
args = parser.parse_args()

print('Téléchargement/chargement du modèle', args.model)
model = YOLO(args.model)
print('Modèle chargé. Chemin interne:', getattr(model.model, 'yaml', 'n/a'))
print("Si l'exécution n'a pas d'erreur, le modèle est disponible localement via ultralytics cache.")

