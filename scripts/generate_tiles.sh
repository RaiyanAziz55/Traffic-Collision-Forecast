#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_GEOJSON="$ROOT_DIR/Data/processed/ottawa_risk_baseline.geojson"
OUTPUT_MBTILES="$ROOT_DIR/Data/processed/ottawa_risk_baseline.mbtiles"
LAYER_NAME="ottawa_risk_baseline"

if [[ ! -f "$INPUT_GEOJSON" ]]; then
  echo "Missing input GeoJSON: $INPUT_GEOJSON" >&2
  exit 1
fi

echo "Generating vector tiles with tippecanoe..."

tippecanoe \
  --force \
  --output="$OUTPUT_MBTILES" \
  --layer="$LAYER_NAME" \
  --maximum-zoom=14 \
  --minimum-zoom=8 \
  --read-parallel \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  "$INPUT_GEOJSON"

echo "Wrote $OUTPUT_MBTILES"
