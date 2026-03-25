import json
import os
from pathlib import Path
from typing import List

import pandas as pd
import requests
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_model.json"
LOOKUP_PATH = BASE_DIR / "road_features_lookup.csv"
METADATA_PATH = BASE_DIR / "model_metadata.json"

DEFAULT_THRESHOLDS = {"yellow": 0.35, "red": 0.55}
DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def parse_cors_origins():
    configured = os.getenv("CORS_ORIGINS")
    if not configured:
        return DEFAULT_CORS_ORIGINS

    return [origin.strip() for origin in configured.split(",") if origin.strip()]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

if LOOKUP_PATH.exists():
    road_feature_lookup = pd.read_csv(LOOKUP_PATH, dtype={"segment_id": str}).set_index("segment_id")
else:
    road_feature_lookup = pd.DataFrame()

if METADATA_PATH.exists():
    model_metadata = json.loads(METADATA_PATH.read_text())
else:
    model_metadata = {"suggested_thresholds": DEFAULT_THRESHOLDS}


class RoadFeature(BaseModel):
    u: int
    v: int
    highway: str | None = None
    length_km: float | None = None
    relative_risk_density: float | None = None


class BatchPredictRequest(BaseModel):
    roads: List[RoadFeature]
    temp: float
    prcp: float
    snow: float
    month: int
    day_of_week: int


class InsightRequest(BaseModel):
    name: str | None = "Unknown Road"
    highway: str
    temp: float
    prcp: float
    snow: float
    historical_count: float
    risk_multiplier: float


def classify_risk(score):
    thresholds = model_metadata.get("suggested_thresholds", DEFAULT_THRESHOLDS)
    if score >= thresholds.get("red", DEFAULT_THRESHOLDS["red"]):
        return "high"
    if score >= thresholds.get("yellow", DEFAULT_THRESHOLDS["yellow"]):
        return "moderate"
    return "lower"


def build_feature_record(road, request_data):
    segment_id = f"{road.u}_{road.v}"
    feature_values = {column: 0.0 for column in model.feature_names_in_}
    feature_values["tavg"] = float(request_data.temp)
    feature_values["prcp"] = float(request_data.prcp)
    feature_values["snow"] = float(request_data.snow)
    feature_values["month"] = int(request_data.month)
    feature_values["day_of_week"] = int(request_data.day_of_week)

    highway_value = road.highway or "unknown"

    if not road_feature_lookup.empty and segment_id in road_feature_lookup.index:
        lookup_row = road_feature_lookup.loc[segment_id]
        highway_value = str(lookup_row.get("highway", highway_value) or "unknown")
        for column, value in lookup_row.items():
            if column == "highway":
                continue
            if column in feature_values:
                feature_values[column] = float(value)
    else:
        fallback_values = {
            "length_km": float(road.length_km or 0.0),
            "relative_risk_density": float(road.relative_risk_density or 0.0),
        }
        for column, value in fallback_values.items():
            if column in feature_values:
                feature_values[column] = value

    highway_column = f"highway_{highway_value}"
    if highway_column in feature_values:
        feature_values[highway_column] = 1.0

    return {"segment_id": segment_id, **feature_values}


@app.get("/model-metadata")
def get_model_metadata():
    return {
        "suggested_thresholds": model_metadata.get("suggested_thresholds", DEFAULT_THRESHOLDS),
        "mean_roc_auc": model_metadata.get("mean_roc_auc"),
        "mean_average_precision": model_metadata.get("mean_average_precision"),
    }


@app.post("/predict-batch")
def predict_batch(data: BatchPredictRequest):
    if not data.roads:
        return {"scores": {}}

    records = [build_feature_record(road, data) for road in data.roads]
    df = pd.DataFrame(records)
    probabilities = model.predict_proba(df[model.feature_names_in_])[:, 1]

    scores = {}
    for index, row in df.iterrows():
        scores[str(row["segment_id"])] = round(float(probabilities[index]), 4)

    return {"scores": scores}


@app.post("/insight")
def generate_insight(data: InsightRequest):
    risk_band = classify_risk(data.risk_multiplier)
    prompt = f"""
    You are an AI traffic safety analyst. Analyze this road segment based on real-time data:
    - Road Name: {data.name} (Type: {data.highway})
    - Live Weather: {data.temp}°C, {data.prcp}mm rain, {data.snow}mm snow.
    - Historical Collisions: {int(data.historical_count)}
    - Current Risk Score: {data.risk_multiplier:.1%}
    - Risk Band: {risk_band}

    Provide a concise, 2-sentence explanation of why this segment is in the stated risk band.
    Do not contradict the risk band.
    Do not call a segment "safe" if the band is moderate or high.
    Keep the explanation grounded in the weather and historical segment context, and avoid robotic language.
    """

    llm_explanation = "LLM Offline or Generating..."
    try:
        ollama_res = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=10,
        )
        if ollama_res.ok:
            llm_explanation = ollama_res.json().get("response", "")
    except Exception as exc:
        print(f"Ollama connection failed: {exc}")

    return {"explanation": llm_explanation}
