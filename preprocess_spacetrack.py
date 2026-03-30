"""
This file is used to preprocess data for spacetrack_starlink.json (SpaceTrack anomaly detection).

I output a file spacetrack_preprocessed.pkl containing the following information that can be loaded for model training and evaluation.
    X_train — scaled feature array (80% of active satellites)
    X_test — scaled feature array (20% held-out)
    feature_names — list of feature column names
    scaler — fitted StandardScaler
    df_train / df_test — original DataFrames (for labeling top anomalies)
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

ORBITAL_FEATURES = [
    "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT", "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS",
    "PERIAPSIS", "REV_AT_EPOCH", "ELEMENT_SET_NO",
]

# Load json file for starlink
with open("spacetrack_starlink.json") as f:
    df = pd.DataFrame(json.load(f))
num_entries = len(df)

# Only used active satellites
df = df[df["DECAY_DATE"].isna()].copy()

# Make orbital features numeric
df[ORBITAL_FEATURES] = df[ORBITAL_FEATURES].apply(pd.to_numeric, errors="coerce")

df = df.dropna(subset=ORBITAL_FEATURES)
dropped = num_entries - len(df)

df["orbit_altitude_range"] = df["APOAPSIS"] - df["PERIAPSIS"]
df["mean_altitude"] = (df["APOAPSIS"] + df["PERIAPSIS"]) / 2.0
df["absolute_motion_dot"] = df["MEAN_MOTION_DOT"].abs()

DERIVED_FEATURES = ["orbit_altitude_range", "mean_altitude", "absolute_motion_dot"]
feature_names = ORBITAL_FEATURES + DERIVED_FEATURES

# Train/test split: 80% train, 20% test
X = df[feature_names].values

X_train_raw, X_test_raw, df_train, df_test = train_test_split(
    X, df, test_size=0.2, random_state=42
)

# Use standard scaler
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

print(f"spacetrack_starlink.json: {num_entries} rows → {len(df)} active clean  "
      f"(dropped {dropped} incomplete)  |  train={len(X_train_raw)}  test={len(X_test_raw)}  "
      f"|  {len(feature_names)} features")

# Adding necessary information to the pickle file
payload = {
    "X_train":      X_train,
    "X_test":       X_test,
    "feature_names": feature_names,
    "scaler":       scaler,
    "df_train":     df_train.reset_index(drop=True),
    "df_test":      df_test.reset_index(drop=True),
}
with open("spacetrack_preprocessed.pkl", "wb") as f:
    pickle.dump(payload, f)

print("New file generated: spacetrack_preprocessed.pkl")
