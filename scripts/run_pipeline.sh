#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$ROOT_DIR/data-pipeline"

echo "Starting offline ML/data pipeline..."
echo

cd "$PIPELINE_DIR"

echo "Step 1/3: Building spatial baseline..."
python3 01_spatial_join.py
echo

echo "Step 2/4: Generating vector tiles..."
"$ROOT_DIR/scripts/generate_tiles.sh"
echo

echo "Step 3/4: Building spatiotemporal training dataset..."
python3 02_build_weather_dataset.py
echo

echo "Step 4/4: Training model and exporting backend artifacts..."
python3 03_train_model.py
echo

echo "Pipeline complete."
