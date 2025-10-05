#!/usr/bin/env bash
# Script d'exemple pour lancer le pipeline en local
if [ -z "$1" ]; then
  echo "Usage: ./run_full_pipeline.sh /chemin/ta_video.mp4 [output_folder]"
  exit 1
fi
VIDEO=$1
OUT=${2:-results}
mkdir -p "$OUT"
python download_models.py
python main.py --video "$VIDEO" --output "$OUT"
