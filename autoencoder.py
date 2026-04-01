"""
Autoencoder for anomaly detection (unsupervised)
- Uses all numeric features
- 20% training, 80% testing
- Threshold computed from training reconstruction errors
- Outputs metrics in autoencoder_results.json
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    accuracy_score
)
import tensorflow as tf
from tensorflow.keras import layers, models

RANDOM_STATE = 42

# Load CSV dataset
csv_path = "dataset.csv"  # change if needed
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

# Select numeric columns (exclude target)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "anomaly"]

# Handle missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Standardize numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Train/test split (20% train, 80% test)
train_df, test_df = train_test_split(
    df,
    train_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["anomaly"]
)

X_train = train_df[numeric_cols].values
y_train = train_df["anomaly"].values

X_test = test_df[numeric_cols].values
y_test = test_df["anomaly"].values

# Build autoencoder
input_dim = X_train.shape[1]
encoding_dim = max(4, input_dim // 2)

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(encoding_dim, activation="relu"),
    layers.Dense(input_dim, activation="linear")
])

autoencoder.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss="mse"
)

# Train autoencoder only on normal points
X_train_normal = X_train[y_train == 0]
autoencoder.fit(
    X_train_normal,
    X_train_normal,
    epochs=50,
    batch_size=32,
    shuffle=True,
    verbose=0
)

# Compute reconstruction errors
recon_train = autoencoder.predict(X_train, verbose=0)
mse_train = np.mean(np.square(X_train - recon_train), axis=1)

# Threshold from training reconstruction errors (95th percentile)
threshold = np.percentile(mse_train, 95)

# Evaluate on test set
recon_test = autoencoder.predict(X_test, verbose=0)
mse_test = np.mean(np.square(X_test - recon_test), axis=1)

y_pred = (mse_test >= threshold).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
pr_auc = average_precision_score(y_test, mse_test)

error_statistics = {
    "mean_mse": float(np.mean(mse_test)),
    "std_mse": float(np.std(mse_test)),
    "threshold": float(threshold),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "pr_auc": float(pr_auc)
}

# Save to autoencoder_results.json
with open("autoencoder_results.json", "w") as f:
    json.dump({"error_statistics": error_statistics}, f, indent=2)

print(json.dumps({"error_statistics": error_statistics}, indent=2))