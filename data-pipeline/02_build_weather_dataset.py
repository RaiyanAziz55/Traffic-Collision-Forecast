from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from meteostat import daily
from shapely.geometry import Point


BASE_DIR = Path(__file__).resolve().parent
RAW_COLLISIONS_PATH = BASE_DIR.parent / "Data" / "raw" / "Traffic_Collision_Data.csv"
BASELINE_ROADS_PATH = BASE_DIR.parent / "Data" / "processed" / "ottawa_risk_baseline.geojson"
OUTPUT_PATH = BASE_DIR.parent / "Data" / "processed" / "spatiotemporal_training_data.csv"

OTTAWA_STATION_ID = "71628"
NEGATIVE_SAMPLES_PER_POSITIVE = 3
NEGATIVE_SAMPLE_ATTEMPTS = 25
SMOOTHING_PRIOR = 5.0
RANDOM_SEED = 42


def normalize_text(series):
    return series.fillna("").astype(str)


def add_smoothed_rate_feature(roads, events, event_counts, flag_column, feature_name):
    global_rate = float(events[flag_column].mean())
    event_hits = events.groupby("segment_id")[flag_column].sum()
    segment_events = roads["segment_id"].map(event_counts).fillna(0.0)
    segment_hits = roads["segment_id"].map(event_hits).fillna(0.0)

    roads[feature_name] = (
        segment_hits + (SMOOTHING_PRIOR * global_rate)
    ) / (segment_events + SMOOTHING_PRIOR)


def add_smoothed_mean_feature(roads, events, event_counts, value_column, feature_name):
    global_mean = float(events[value_column].fillna(0).mean())
    value_sums = events.groupby("segment_id")[value_column].sum(min_count=1)
    segment_events = roads["segment_id"].map(event_counts).fillna(0.0)
    segment_sums = roads["segment_id"].map(value_sums).fillna(0.0)

    roads[feature_name] = (
        segment_sums + (SMOOTHING_PRIOR * global_mean)
    ) / (segment_events + SMOOTHING_PRIOR)


def sample_negative_date(rng, crash_dates, primary_pool, month_pool, all_dates, used_dates):
    for pool in (primary_pool, month_pool, all_dates):
        if pool is None or len(pool) == 0:
            continue

        for _ in range(NEGATIVE_SAMPLE_ATTEMPTS):
            candidate = pd.Timestamp(rng.choice(pool))
            if candidate in crash_dates or candidate in used_dates:
                continue
            return candidate

    for candidate_raw in all_dates:
        candidate = pd.Timestamp(candidate_raw)
        if candidate in crash_dates or candidate in used_dates:
            continue
        return candidate

    raise RuntimeError("Unable to sample a negative date for the requested road segment.")


print("1. Loading raw collisions and the spatial baseline...")
df = pd.read_csv(RAW_COLLISIONS_PATH)
df = df.dropna(subset=["Long", "Lat", "Accident_Date"])
df["Accident_Date"] = pd.to_datetime(df["Accident_Date"], errors="coerce").dt.normalize()
df = df.dropna(subset=["Accident_Date"])

geometry = [Point(xy) for xy in zip(df["Long"], df["Lat"])]
collisions = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=32618)

roads = gpd.read_file(BASELINE_ROADS_PATH).to_crs(epsg=32618)
roads["segment_id"] = roads["u"].astype(str) + "_" + roads["v"].astype(str)
roads["name_key"] = roads["name"].fillna("Unknown Road").astype(str)

segment_midpoints = roads.geometry.centroid
roads["segment_mid_x"] = segment_midpoints.x
roads["segment_mid_y"] = segment_midpoints.y

node_risk = pd.concat(
    [
        roads[["u", "relative_risk_density"]].rename(columns={"u": "node_id"}),
        roads[["v", "relative_risk_density"]].rename(columns={"v": "node_id"}),
    ],
    ignore_index=True,
).groupby("node_id")["relative_risk_density"].mean()

node_degree = pd.concat(
    [
        roads[["u"]].rename(columns={"u": "node_id"}),
        roads[["v"]].rename(columns={"v": "node_id"}),
    ],
    ignore_index=True,
).groupby("node_id").size()

roads["road_name_mean_relative_risk"] = roads.groupby("name_key")["relative_risk_density"].transform("mean")
roads["connected_node_risk"] = (
    roads["u"].map(node_risk).fillna(roads["relative_risk_density"])
    + roads["v"].map(node_risk).fillna(roads["relative_risk_density"])
) / 2.0
roads["connected_node_degree"] = (
    roads["u"].map(node_degree).fillna(1.0) + roads["v"].map(node_degree).fillna(1.0)
) / 2.0

print("2. Snapping collisions to the road network...")
snapped = gpd.sjoin_nearest(collisions, roads, how="inner", max_distance=50)
snapped["segment_id"] = snapped["u"].astype(str) + "_" + snapped["v"].astype(str)
snapped["date"] = snapped["Accident_Date"].dt.normalize()

print("3. Engineering historical road features from the collision CSV...")
location = normalize_text(snapped["Location"])
traffic_control = normalize_text(snapped["Traffic_Control"])
surface = normalize_text(snapped["Road_1_Surface_Condition"])
environment = normalize_text(snapped["Environment_Condition_1"])
light = normalize_text(snapped["Light"])
impact = normalize_text(snapped["Initial_Impact_Type"])
classification = normalize_text(snapped["Classification_Of_Accident"])
max_injury = normalize_text(snapped["Max_injury"])

snapped["is_intersection_collision"] = location.str.contains("@", regex=False)
snapped["has_traffic_signal"] = traffic_control.str.startswith("01 - Traffic signal")
snapped["has_stop_sign"] = traffic_control.str.startswith("02 - Stop sign")
snapped["has_no_control"] = traffic_control.str.startswith("10 - No control")
snapped["has_roundabout"] = traffic_control.str.startswith("11 - Roundabout")

snapped["wet_surface_collision"] = surface.str.startswith("02 - Wet")
snapped["snow_surface_collision"] = surface.str.startswith(
    ("03 - Loose snow", "04 - Slush", "05 - Packed snow")
)
snapped["ice_surface_collision"] = surface.str.startswith("06 - Ice")

snapped["rain_environment_collision"] = environment.str.startswith("02 - Rain")
snapped["snow_environment_collision"] = environment.str.startswith(
    ("03 - Snow", "05 - Drifting Snow")
)
snapped["freezing_rain_collision"] = environment.str.startswith("04 - Freezing Rain")
snapped["dark_condition_collision"] = light.str.contains("Dark|Dusk|Dawn", regex=True)

snapped["pedestrian_collision"] = snapped["num_of_pedestrians"].fillna(0).gt(0)
snapped["bicycle_collision"] = snapped["num_of_bicycles"].fillna(0).gt(0)
snapped["motorcycle_collision"] = snapped["num_of_motorcycles"].fillna(0).gt(0)
snapped["multi_vehicle_collision"] = snapped["num_of_vehicles"].fillna(0).gt(1)

snapped["injury_collision"] = classification.str.startswith(
    ("01 - Fatal injury", "02 - Non-fatal injury")
) | snapped["num_of_injuries"].fillna(0).gt(0)
snapped["major_or_fatal_collision"] = max_injury.isin(["Major", "Fatal"]) | classification.str.startswith(
    "01 - Fatal injury"
)

snapped["rear_end_collision"] = impact.str.startswith("03 - Rear end")
snapped["angle_collision"] = impact.str.startswith("02 - Angle")
snapped["turning_collision"] = impact.str.startswith("05 - Turning movement")

event_counts = snapped.groupby("segment_id").size()
crash_day_counts = snapped.groupby("segment_id")["date"].nunique()

roads["historical_event_count_log1p"] = np.log1p(roads["segment_id"].map(event_counts).fillna(0.0))
roads["historical_crash_day_count_log1p"] = np.log1p(
    roads["segment_id"].map(crash_day_counts).fillna(0.0)
)

rate_features = {
    "is_intersection_collision": "intersection_collision_rate",
    "has_traffic_signal": "traffic_signal_rate",
    "has_stop_sign": "stop_sign_rate",
    "has_no_control": "no_control_rate",
    "has_roundabout": "roundabout_rate",
    "wet_surface_collision": "wet_surface_rate",
    "snow_surface_collision": "snow_surface_rate",
    "ice_surface_collision": "ice_surface_rate",
    "rain_environment_collision": "rain_environment_rate",
    "snow_environment_collision": "snow_environment_rate",
    "freezing_rain_collision": "freezing_rain_rate",
    "dark_condition_collision": "dark_condition_rate",
    "pedestrian_collision": "pedestrian_collision_rate",
    "bicycle_collision": "bicycle_collision_rate",
    "motorcycle_collision": "motorcycle_collision_rate",
    "multi_vehicle_collision": "multi_vehicle_rate",
    "injury_collision": "injury_collision_rate",
    "major_or_fatal_collision": "major_or_fatal_rate",
    "rear_end_collision": "rear_end_rate",
    "angle_collision": "angle_rate",
    "turning_collision": "turning_rate",
}

for flag_column, feature_name in rate_features.items():
    add_smoothed_rate_feature(roads, snapped, event_counts, flag_column, feature_name)

add_smoothed_mean_feature(roads, snapped, event_counts, "num_of_vehicles", "avg_num_vehicles")

print("4. Building a road-day target matrix...")
positives = (
    snapped.groupby(["u", "v", "segment_id", "date"], as_index=False)
    .agg(daily_collision_count=("ID", "size"))
    .sort_values(["segment_id", "date"])
)
positives["target"] = 1
positives["month"] = positives["date"].dt.month
positives["day_of_week"] = positives["date"].dt.dayofweek

print("5. Generating temporally matched negative samples...")
min_date = positives["date"].min()
max_date = positives["date"].max()

calendar = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D")})
calendar["month"] = calendar["date"].dt.month
calendar["day_of_week"] = calendar["date"].dt.dayofweek

month_day_lookup = {
    key: value["date"].to_numpy(dtype="datetime64[ns]")
    for key, value in calendar.groupby(["month", "day_of_week"])
}
month_lookup = {
    key: value["date"].to_numpy(dtype="datetime64[ns]")
    for key, value in calendar.groupby("month")
}
all_dates = calendar["date"].to_numpy(dtype="datetime64[ns]")

crash_dates_by_segment = positives.groupby("segment_id")["date"].apply(set).to_dict()
rng = np.random.default_rng(RANDOM_SEED)
negative_rows = []

for row in positives.itertuples(index=False):
    crash_dates = crash_dates_by_segment[row.segment_id]
    used_dates = set()

    primary_pool = month_day_lookup.get((row.month, row.day_of_week))
    month_pool = month_lookup.get(row.month)

    for _ in range(NEGATIVE_SAMPLES_PER_POSITIVE):
        negative_date = sample_negative_date(
            rng,
            crash_dates=crash_dates,
            primary_pool=primary_pool,
            month_pool=month_pool,
            all_dates=all_dates,
            used_dates=used_dates,
        )
        used_dates.add(negative_date)
        negative_rows.append(
            {
                "u": row.u,
                "v": row.v,
                "segment_id": row.segment_id,
                "date": negative_date,
                "daily_collision_count": 0,
                "target": 0,
            }
        )

negatives = pd.DataFrame(negative_rows)
negatives["month"] = negatives["date"].dt.month
negatives["day_of_week"] = negatives["date"].dt.dayofweek

ml_data = pd.concat(
    [
        positives[
            ["u", "v", "segment_id", "date", "daily_collision_count", "target", "month", "day_of_week"]
        ],
        negatives[
            ["u", "v", "segment_id", "date", "daily_collision_count", "target", "month", "day_of_week"]
        ],
    ],
    ignore_index=True,
)

print("6. Fetching historical Ottawa weather data...")
weather = daily(OTTAWA_STATION_ID, min_date.to_pydatetime(), max_date.to_pydatetime()).fetch()
if weather is None or weather.empty:
    raise ValueError("Meteostat pulled empty data. Check the cached data or network access.")

weather = weather.reset_index()
weather["date"] = pd.to_datetime(weather["time"]).dt.normalize()
weather = weather.rename(columns={"temp": "tavg", "snwd": "snow"})
weather_features = weather[["date", "tavg", "prcp", "snow"]].fillna(0)

print("7. Merging weather and road features...")
road_feature_columns = [
    "u",
    "v",
    "segment_id",
    "highway",
    "length_km",
    "relative_risk_density",
    "road_name_mean_relative_risk",
    "connected_node_risk",
    "connected_node_degree",
    "segment_mid_x",
    "segment_mid_y",
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

ml_data = ml_data.merge(weather_features, on="date", how="left")
ml_data = ml_data.merge(roads[road_feature_columns], on=["u", "v", "segment_id"], how="left")

for weather_column in ["tavg", "prcp", "snow"]:
    ml_data[weather_column] = ml_data[weather_column].fillna(0)

print("8. Exporting the improved training matrix...")
ml_data.to_csv(OUTPUT_PATH, index=False)

print(
    "Success! Built an improved road-day matrix with "
    f"{len(positives)} positive road-days and {len(negatives)} negative road-days."
)
