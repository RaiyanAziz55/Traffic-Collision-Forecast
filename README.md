# Traffic Collision Forecast

This project is a context-aware traffic risk forecasting system for Ottawa road segments. It combines historical collision data, weather-aware machine learning, vector tiles, interactive map-based inference, and local AI explanations.

## What Runs in Docker

The Docker setup includes:

- `frontend`: React + MapLibre web app
- `backend`: FastAPI inference API
- `martin`: vector tile server for the road network
- `ollama`: local LLM service for AI explanations
- `pipeline`: optional offline retraining container

The app stack and the retraining pipeline are intentionally separate.

---

## Quickstart

### 1. Prerequisites

Install:

- Docker Desktop

### 2. Run the full setup

From the project root:

```bash
./scripts/bootstrap.sh
```

This script does the full first-run flow in the correct order:

1. runs the offline pipeline
2. generates the vector tiles and model artifacts
3. starts the Docker app stack
4. ensures `llama3.2` is installed in Ollama
5. restarts the backend

When it finishes, the following are available:

- frontend on `http://localhost:8080`
- backend on `http://localhost:8000`
- tile server on `http://localhost:3000`
- Ollama on `http://localhost:11434`

### 3. Open the app

Open:

```text
http://localhost:8080
```

### 4. Test the full flow

To verify the app end to end:

1. wait for the map and weather panel to load
2. click a neighborhood to score roads
3. click a colored road to open the side panel
4. confirm the AI assessment appears

---

## First-Time Setup Notes

### Why AI explanations may not work immediately

Ollama does not ship with models preinstalled. If you skip the model pull step, the app still loads and road scoring still works, but AI explanations will stay unavailable.

### How `localhost` becomes `ollama`

Inside Docker, `localhost` means the current container only. That is why the backend cannot use `http://localhost:11434` to reach Ollama.

Inside the Docker network, services talk to each other by service name:

- backend -> `http://ollama:11434/api`
- browser -> `http://localhost:8080`

The backend is configured through:

```text
OLLAMA_BASE_URL=http://ollama:11434/api
```

When the backend is run outside Docker, it falls back to:

```text
http://localhost:11434/api
```

---

## Running the Offline ML/Data Pipeline

Use this only when you want to rebuild the baseline, regenerate tiles, and retrain the model.

Run:

```bash
docker compose --profile pipeline run --rm pipeline
```

This executes the full offline workflow:

1. `01_spatial_join.py`
2. `scripts/generate_tiles.sh`
3. `02_build_weather_dataset.py`
4. `03_train_model.py`

Outputs are written back into the repo, including:

- `Data/processed/ottawa_risk_baseline.geojson`
- `Data/processed/ottawa_risk_baseline.mbtiles`
- `Data/processed/spatiotemporal_training_data.csv`
- `backend/xgb_model.json`
- `backend/road_features_lookup.csv`
- `backend/model_metadata.json`

---

## Common Commands

Start everything:

```bash
docker compose up --build -d
```

Run the full first-time setup:

```bash
./scripts/bootstrap.sh
```

Watch logs:

```bash
docker compose logs -f
```

Watch one service:

```bash
docker compose logs -f backend
docker compose logs -f ollama
```

Stop everything:

```bash
docker compose down
```

Stop and remove Ollama model volume too:

```bash
docker compose down -v
```

Check running services:

```bash
docker compose ps
```

List installed Ollama models:

```bash
docker compose exec ollama ollama list
```

Run a quick Ollama test:

```bash
docker compose exec ollama ollama run llama3.2 "Give a one-sentence traffic safety explanation."
```

Restart the backend:

```bash
docker compose restart backend
```

Run the offline pipeline:

```bash
docker compose --profile pipeline run --rm pipeline
```

---

## Troubleshooting

### The app loads but AI assessment says "LLM Offline or Generating..."

Check that Ollama is running:

```bash
docker compose ps
```

Check that the model is installed:

```bash
docker compose exec ollama ollama list
```

If `llama3.2` is missing, install it:

```bash
docker compose exec ollama ollama pull llama3.2
docker compose restart backend
```

### `service "ollama" is not running`

Start it explicitly:

```bash
docker compose up -d ollama
```

Then verify:

```bash
docker compose ps
docker compose logs ollama
```

### The model pulls successfully but AI still does not appear

Restart the backend:

```bash
docker compose restart backend
```

Then test Ollama directly:

```bash
docker compose exec ollama ollama run llama3.2 "Say hello in one sentence."
```

### The pipeline runs but tile serving fails

The app expects:

- `Data/processed/ottawa_risk_baseline.mbtiles`

If that file is missing, rerun:

```bash
docker compose --profile pipeline run --rm pipeline
```

### Full reset

If Docker state gets messy:

```bash
docker compose down -v
./scripts/bootstrap.sh
```

---

## Notes

- The frontend uses `VITE_API_BASE_URL=http://localhost:8000`
- The frontend uses `VITE_TILE_URL=http://localhost:3000/ottawa_risk_baseline`
- Martin serves the prebuilt `.mbtiles` file generated by the offline pipeline
- Tile generation is included in the pipeline through `tippecanoe`
- The pipeline container is a one-off batch job, not a long-running service
- `./scripts/bootstrap.sh` is the recommended setup path because it enforces the correct order
