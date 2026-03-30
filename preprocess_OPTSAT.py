"""
This file is used to preprocess data for OPTSAT_dataset.csv (contains OPTSAT-AD data). It outputs a pickle file named optsat_preprocessed.pkl that contains the following:
- X_train: 2D numpy array of training features (scaled)
- X_test: 2D numpy array of test features (scaled)
- y_train: 1D numpy array of training labels (0 or 1)
- y_test: 1D numpy array of test labels (0 or 1)
- feature_names: list of feature names corresponding to columns in X_train/X_test
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("OPTSAT_dataset.csv")
value_counts = df["anomaly"].value_counts()
print(f"OPTSAT_dataset.csv: {len(df)} rows  |  anomaly={int(value_counts.get(1, 0))}  normal={int(value_counts.get(0, 0))}  "
      f"|  {df['channel'].nunique()} channels")

# One-hot encode channel
channel_dummies = pd.get_dummies(df["channel"], prefix="ch").astype(int)
channel_cols    = list(channel_dummies.columns)

# Sampling rate as a numeric feature
df["sampling_rate"] = df["sampling"]

# Signal statistic features (same as before)
DROP_COLS = ["segment", "channel", "sampling", "anomaly", "train"]
signal_features = [c for c in df.columns if c not in DROP_COLS + ["sampling_rate"]]

# Combine: signal stats + sampling rate + channel dummies
df_features = pd.concat([
    df[signal_features + ["sampling_rate"]],
    channel_dummies,
], axis=1)

feature_names = list(df_features.columns)

# Train/test split using train attribute (1 = train, 0 = test)
train_mask  = df["train"] == 1
X_train_raw = df_features.loc[train_mask].values
X_test_raw  = df_features.loc[~train_mask].values
y_train     = df.loc[train_mask, "anomaly"].values
y_test      = df.loc[~train_mask, "anomaly"].values

# Preserve channel labels and row ids
channel_train = df.loc[train_mask, "channel"].values
channel_test  = df.loc[~train_mask, "channel"].values
segment_test  = df.loc[~train_mask, "segment"].values
sampling_test = df.loc[~train_mask, "sampling"].values

print(f"Split: train={len(y_train)} (anomaly {y_train.mean():.1%})  "
      f"test={len(y_test)} (anomaly {y_test.mean():.1%})  |  {len(feature_names)} features")

# Using standard scaler
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# Save preprocessed data to pickle file
payload = {
    "X_train":       X_train,
    "X_test":        X_test,
    "y_train":       y_train,
    "y_test":        y_test,
    "feature_names": feature_names,
    "scaler":        scaler,
    "channel_train": channel_train,
    "channel_test":  channel_test,
    "channel_cols":  channel_cols,
    "segment_test":  segment_test,
    "sampling_test": sampling_test,
}
with open("optsat_preprocessed.pkl", "wb") as f:
    pickle.dump(payload, f)

print("New generated file: optsat_preprocessed.pkl")
