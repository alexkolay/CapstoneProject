from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def engineer_features(df):
    # Orbital energy / size
    df['semimajor_axis']   = df['SEMIMAJOR_AXIS'].astype(float)
    df['period']           = df['PERIOD'].astype(float)
    df['mean_motion']      = df['MEAN_MOTION'].astype(float)

    # Orbit shape
    df['eccentricity']     = df['ECCENTRICITY'].astype(float)

    # Orbit orientation (stable under normal ops)
    df['inclination']      = df['INCLINATION'].astype(float)

    # Drag and decay signals
    df['bstar']            = df['BSTAR'].astype(float)
    df['mm_dot']           = df['MEAN_MOTION_DOT'].astype(float)
    df['mm_ddot']          = df['MEAN_MOTION_DDOT'].astype(float)

    # Altitude-derived features
    df['apoapsis']         = df['APOAPSIS'].astype(float)
    df['periapsis']        = df['PERIAPSIS'].astype(float)
    df['altitude_spread']  = df['apoapsis'] - df['periapsis']  # orbit circularization

    # Drop fields that change constantly from normal mechanics
    # RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY — not useful here

    return df

features = [
    'semimajor_axis', 'eccentricity', 'inclination',
    'bstar', 'mm_dot', 'mm_ddot',
    'apoapsis', 'periapsis', 'altitude_spread'
]


df = pd.read_json('spacetrack_starlink.json')

df = engineer_features(df)
X = df[features].copy()

# --- Step 1: Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 2: Cluster by orbit regime ---
# LEO, MEO, GEO, HEO will naturally separate on semimajor_axis + inclination
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(X_scaled)

# --- Step 3: Fit an outlier detector WITHIN each cluster ---
# This is key — you're asking "is this weird for its peer group"
# not "is this weird globally"
iso_forests = {}
df['anomaly_score'] = 0.0
df['is_anomaly']    = False

for cluster_id in df['cluster'].unique():
    mask = df['cluster'] == cluster_id
    X_cluster = X_scaled[mask]

    iso = IsolationForest(
        contamination=0.05,  # expect ~5% anomalies per cluster
        random_state=42,
        n_estimators=100
    )
    iso.fit(X_cluster)
    iso_forests[cluster_id] = iso

    # score_samples returns negative values — more negative = more anomalous
    df.loc[mask, 'anomaly_score'] = -iso.score_samples(X_cluster)
    df.loc[mask, 'is_anomaly']    = iso.predict(X_cluster) == -1

# --- Step 4: Inspect flagged satellites ---
anomalies = df[df['is_anomaly']].sort_values('anomaly_score', ascending=False)
print(anomalies[['OBJECT_NAME', 'cluster', 'anomaly_score'] + features])


print(df[features].describe())
print(df['cluster'].value_counts())


print(df.groupby('cluster')['inclination'].describe())
print(df.groupby('cluster')['semimajor_axis'].describe())