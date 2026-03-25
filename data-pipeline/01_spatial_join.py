import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import Point
import numpy as np

BAYES_PRIOR_LENGTH_KM = 0.25

print("1. Downloading Ottawa road network from OpenStreetMap...")
# "drive" ensures we only get roads cars can drive on
G = ox.graph_from_place("Ottawa, Ontario, Canada", network_type="drive")
nodes, edges = ox.graph_to_gdfs(G)

# Reproject to UTM Zone 18N (meters) for accurate physical distance calculations
edges = edges.to_crs(epsg=32618) 

print("2. Loading collision data...")
# Read the CSV from your new structured directory
df = pd.read_csv("../Data/raw/Traffic_Collision_Data.csv")

# Use 'Long' and 'Lat' as defined in the Ottawa dataset
df = df.dropna(subset=['Long', 'Lat'])
geometry = [Point(xy) for xy in zip(df['Long'], df['Lat'])]
collisions = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Reproject collisions to match the road network's meter-based projection
collisions = collisions.to_crs(epsg=32618)

print("3. Snapping collisions to the nearest road segment...")
snapped = gpd.sjoin_nearest(collisions, edges, how="left", max_distance=50)

print("4. Aggregating counts and engineering Risk Density...")
collision_counts = snapped.groupby(['u', 'v']).size().reset_index(name='historical_count')
edges_with_data = edges.merge(collision_counts, on=['u', 'v'], how='left')
edges_with_data['historical_count'] = edges_with_data['historical_count'].fillna(0)

# Clean up the OSM highway tags (fixing the list/array issue right at the source)
edges_with_data['highway'] = edges_with_data['highway'].apply(
    lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
)
edges_with_data['highway'] = edges_with_data['highway'].fillna('unknown').astype(str)

# Calculate physical length in kilometers
edges_with_data['length_km'] = edges_with_data.geometry.length / 1000.0
edges_with_data['length_km'] = edges_with_data['length_km'].replace(0, 0.001) # Prevent division by zero

# Calculate raw density: Collisions per kilometer
edges_with_data['raw_collisions_per_km'] = edges_with_data['historical_count'] / edges_with_data['length_km']

print("5. Stratifying Relative Risk by Road Class...")
# Calculate the average density for each specific type of road across all of Ottawa
class_averages = edges_with_data.groupby('highway')['raw_collisions_per_km'].mean().reset_index()
class_averages.rename(columns={'raw_collisions_per_km': 'class_avg_density'}, inplace=True)

edges_with_data = edges_with_data.merge(class_averages, on='highway', how='left')

# Smooth the density for short segments so one crash on a tiny edge does not dominate the map.
edges_with_data['collisions_per_km'] = (
    edges_with_data['historical_count'] + (edges_with_data['class_avg_density'] * BAYES_PRIOR_LENGTH_KM)
) / (edges_with_data['length_km'] + BAYES_PRIOR_LENGTH_KM)

# Calculate Relative Risk: $Relative Risk = \frac{Segment Density}{Class Average Density}$
# A score of 1.0 means it is exactly average for its road type. 
# A score of 5.0 means it is 5x more dangerous than other roads of its type.
edges_with_data['relative_risk_density'] = edges_with_data['collisions_per_km'] / (edges_with_data['class_avg_density'] + 0.01)

print("6. Exporting to GeoJSON...")
columns_to_keep = [
    'u', 'v', 'name', 'highway', 'length_km', 'historical_count', 
    'raw_collisions_per_km', 'collisions_per_km', 'class_avg_density', 'relative_risk_density', 'geometry'
]
final_gdf = edges_with_data[[c for c in columns_to_keep if c in edges_with_data.columns]]

# Revert to standard Lat/Lon for MapLibre web mapping
final_gdf = final_gdf.to_crs("EPSG:4326")
final_gdf.to_file("../Data/processed/ottawa_risk_baseline.geojson", driver="GeoJSON")

print("Success! Spatially normalized baseline is ready.")
