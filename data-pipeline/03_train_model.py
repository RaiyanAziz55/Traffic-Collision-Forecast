import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR.parent / "Data" / "processed" / "spatiotemporal_training_data.csv"
BACKEND_DIR = BASE_DIR.parent / "backend"
MODEL_PATH = BACKEND_DIR / "xgb_model.json"
LOOKUP_PATH = BACKEND_DIR / "road_features_lookup.csv"
METADATA_PATH = BACKEND_DIR / "model_metadata.json"

SPATIAL_BLOCK_SIZE_METERS = 2000
RANDOM_STATE = 42

TIME_AND_WEATHER_COLUMNS = {"tavg", "prcp", "snow", "month", "day_of_week"}
NON_FEATURE_COLUMNS = {
    "u",
    "v",
    "segment_id",
    "date",
    "target",
    "daily_collision_count",
    "segment_mid_x",
    "segment_mid_y",
}


print("1. Loading the improved road-day matrix...")
df = pd.read_csv(TRAINING_DATA_PATH)
df["segment_id"] = df["u"].astype(str) + "_" + df["v"].astype(str)

print("2. Building the backend road lookup...")
road_lookup_columns = [
    "segment_id",
    "u",
    "v",
    "highway",
    "length_km",
    "relative_risk_density",
    "road_name_mean_relative_risk",
    "connected_node_risk",
    "connected_node_degree",
    "historical_event_count_log1p",
    "historical_crash_day_count_log1p",
    "intersection_collision_rate",
    "traffic_signal_rate",
    "stop_sign_rate",
    "no_control_rate",
    "roundabout_rate",
    "wet_surface_rate",
    "snow_surface_rate",
    "ice_surface_rate",
    "rain_environment_rate",
    "snow_environment_rate",
    "freezing_rain_rate",
    "dark_condition_rate",
    "pedestrian_collision_rate",
    "bicycle_collision_rate",
    "motorcycle_collision_rate",
    "multi_vehicle_rate",
    "injury_collision_rate",
    "major_or_fatal_rate",
    "rear_end_rate",
    "angle_rate",
    "turning_rate",
    "avg_num_vehicles",
]

road_lookup = df[road_lookup_columns].drop_duplicates(subset=["segment_id"]).copy()
BACKEND_DIR.mkdir(exist_ok=True)
road_lookup.to_csv(LOOKUP_PATH, index=False)

print("3. Preparing features for model training...")
feature_columns = [
    column
    for column in df.columns
    if column not in NON_FEATURE_COLUMNS
]

y = df["target"]
X = df[feature_columns].copy()
X["highway"] = X["highway"].fillna("unknown").astype(str)
X = pd.get_dummies(X, columns=["highway"])

numeric_columns = X.columns
X[numeric_columns] = X[numeric_columns].fillna(0)

grid_x = np.floor(df["segment_mid_x"] / SPATIAL_BLOCK_SIZE_METERS).astype(int)
grid_y = np.floor(df["segment_mid_y"] / SPATIAL_BLOCK_SIZE_METERS).astype(int)
groups = grid_x.astype(str) + "_" + grid_y.astype(str)

print("4. Running block spatial cross-validation...")
n_splits = min(5, groups.nunique())
if n_splits < 2:
    raise ValueError("Not enough spatial blocks were generated for cross-validation.")

gkf = GroupKFold(n_splits=n_splits)
fold_metrics = []

model_params = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}

for fold_number, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scale_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    model = xgb.XGBClassifier(scale_pos_weight=scale_weight, **model_params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    fold_auc = roc_auc_score(y_test, y_prob)
    fold_ap = average_precision_score(y_test, y_prob)
    fold_metrics.append({"fold": fold_number, "roc_auc": fold_auc, "average_precision": fold_ap})

    print(
        f"Fold {fold_number}: ROC-AUC={fold_auc:.4f}, "
        f"Average Precision={fold_ap:.4f}, test rows={len(test_idx)}"
    )

print("5. Training the final model on the full dataset...")
final_scale_weight = (len(y) - y.sum()) / max(y.sum(), 1)
final_model = xgb.XGBClassifier(scale_pos_weight=final_scale_weight, **model_params)
final_model.fit(X, y)
final_model.save_model(MODEL_PATH)

train_probabilities = final_model.predict_proba(X)[:, 1]
metadata = {
    "feature_columns": list(X.columns),
    "spatial_block_size_meters": SPATIAL_BLOCK_SIZE_METERS,
    "cv_metrics": fold_metrics,
    "mean_roc_auc": float(np.mean([metric["roc_auc"] for metric in fold_metrics])),
    "mean_average_precision": float(np.mean([metric["average_precision"] for metric in fold_metrics])),
    "suggested_thresholds": {
        "yellow": float(np.quantile(train_probabilities, 0.75)),
        "red": float(np.quantile(train_probabilities, 0.9)),
    },
}

METADATA_PATH.write_text(json.dumps(metadata, indent=2))

print("6. Top learned features...")
feature_importances = sorted(
    zip(final_model.feature_names_in_, final_model.feature_importances_),
    key=lambda item: item[1],
    reverse=True,
)
for feature_name, importance in feature_importances[:15]:
    print(f"{feature_name}: {importance:.4f}")

print(
    "Success! Saved the retrained model, backend road lookup, and metadata. "
    f"Mean ROC-AUC={metadata['mean_roc_auc']:.4f}, "
    f"Mean AP={metadata['mean_average_precision']:.4f}."
)
