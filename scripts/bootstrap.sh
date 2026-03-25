#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${OLLAMA_MODEL:-llama3.2}"

cd "$ROOT_DIR"

echo "Step 1/4: Running the offline pipeline..."
docker compose --profile pipeline run --rm pipeline
echo

echo "Step 2/4: Starting the app stack..."
docker compose up --build -d
echo

echo "Step 3/4: Ensuring Ollama model '$MODEL_NAME' is installed..."
if docker compose exec ollama ollama list | awk 'NR > 1 {print $1}' | grep -qx "$MODEL_NAME"; then
  echo "Model '$MODEL_NAME' is already installed."
else
  docker compose exec ollama ollama pull "$MODEL_NAME"
fi
echo

echo "Step 4/4: Restarting the backend..."
docker compose restart backend
echo

echo "Setup complete."
echo "Open http://localhost:8080"
