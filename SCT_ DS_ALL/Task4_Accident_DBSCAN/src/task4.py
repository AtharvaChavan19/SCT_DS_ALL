import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ======================
# CONFIGURATION
# ======================
CSV_FILE = "../data/US_Accidents_March23.csv"
MAX_POINTS = 100_000  
EPS_KM = 0.5         
MIN_SAMPLES = 50      

# ======================
# LOAD DATA
# ======================
print(f"Loading dataset from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

print(f"Original dataset size: {len(df):,} rows")
if len(df) > MAX_POINTS:
    print(f"Sampling {MAX_POINTS:,} rows to prevent MemoryError...")
    df = df.sample(MAX_POINTS, random_state=42)

# Drop rows without coordinates
if not {'Start_Lat', 'Start_Lng'}.issubset(df.columns):
    raise ValueError("Dataset must have 'Start_Lat' and 'Start_Lng' columns")

df = df.dropna(subset=['Start_Lat', 'Start_Lng'])
print(f"Data after dropping NaN coords: {len(df):,} rows")

# ======================
# DBSCAN CLUSTERING
# ======================
coords = df[['Start_Lat', 'Start_Lng']].values

db = DBSCAN(
    eps=EPS_KM / 6371.0,  
    min_samples=MIN_SAMPLES,
    algorithm='ball_tree',
    metric='haversine'
).fit(np.radians(coords))

df['Cluster'] = db.labels_
num_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)
print(f"Found {num_clusters} clusters (excluding noise)")

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(10, 6))
clusters = df['Cluster'].unique()
for i, cluster in enumerate(clusters):
    cluster_points = df[df['Cluster'] == cluster]
    color = 'gray' if cluster == -1 else None
    label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
    plt.scatter(
        cluster_points['Start_Lat'], cluster_points['Start_Lng'],
        s=1, c=color, label=label if i < 10 else None, alpha=0.5
    )

plt.title("Accident Hotspots (DBSCAN Clustering)")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.legend(markerscale=5, fontsize='small', loc='best')
plt.show()

# ======================
# SAVE RESULTS
# ======================
df.to_csv("accident_clusters.csv", index=False)
print("Clustered data saved to accident_clusters.csv")

# ======================
# TIME-BASED ANALYSIS
# ======================
df['Hour'] = pd.to_datetime(df['Start_Time'], errors='coerce').dt.hour
hour_counts = df['Hour'].value_counts().sort_index()
plt.figure(figsize=(10,5))
hour_counts.plot(kind='bar', color='orange')
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

df['DayOfWeek'] = pd.to_datetime(df['Start_Time'], errors='coerce').dt.dayofweek 
dow_counts = df['DayOfWeek'].value_counts().sort_index()
plt.figure(figsize=(8,4))
dow_counts.plot(kind='bar', color='green')
plt.title("Accidents by Day of Week")
plt.xlabel("Day of Week (Mon=0)")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# ======================
# WEATHER ANALYSIS
# ======================
if 'Weather_Condition' in df.columns:
    weather_counts = df['Weather_Condition'].value_counts().head(10)
    plt.figure(figsize=(10,5))
    weather_counts.plot(kind='barh', color='skyblue')
    plt.title("Top 10 Weather Conditions at Accident Time")
    plt.xlabel("Number of Accidents")
    plt.ylabel("Weather Condition")
    plt.tight_layout()
    plt.show()

# ======================
# HEATMAP: Day vs Hour
# ======================
pivot_value_col = 'ID' if 'ID' in df.columns else 'Start_Time'
pivot = df.pivot_table(index='DayOfWeek', columns='Hour', values=pivot_value_col,
                       aggfunc='count', fill_value=0)

plt.figure(figsize=(12,6))
plt.imshow(pivot, aspect='auto', cmap='YlOrRd')
plt.colorbar(label='Number of Accidents')
plt.xticks(range(24), range(24))
plt.yticks(range(7), ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.title("Accidents Heatmap: Day vs Hour")
plt.tight_layout()
plt.show()
