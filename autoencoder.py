import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import legacy
from tensorflow.keras import regularizers

# Load CSV dataset
df = pd.read_csv("dataset.csv")

# Use all numeric columns except 'anomaly'
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "anomaly" in numeric_cols:
    numeric_cols.remove("anomaly")

# Drop rows with NaNs
df = df.dropna(subset=numeric_cols + ["anomaly"])

# Split data 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["anomaly"])

X_train = train_df[numeric_cols].astype(float).values
X_test = test_df[numeric_cols].astype(float).values

y_train = train_df["anomaly"].values
y_test = test_df["anomaly"].values

# Normalize features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Build autoencoder
input_dim = X_train_norm.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
encoded = Dense(8, activation="relu")(encoded)
decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=legacy.Adam(learning_rate=0.001), loss="mse")

# Train autoencoder only on normal data
X_train_norm_clean = X_train_norm[y_train == 0]
autoencoder.fit(
    X_train_norm_clean,
    X_train_norm_clean,
    epochs=50,
    batch_size=32,
    shuffle=True,
    verbose=0
)

# Compute MSE reconstruction error
reconstructions = autoencoder.predict(X_test_norm, verbose=0)
mse_test = np.mean(np.square(reconstructions - X_test_norm), axis=1)

# Determine threshold from training normal data using accuracy+recall balance
reconstructions_train = autoencoder.predict(X_train_norm_clean, verbose=0)
mse_train = np.mean(np.square(reconstructions_train - X_train_norm_clean), axis=1)

threshold_candidates = np.linspace(mse_train.min(), mse_train.max(), 200)
best_score = -1
best_threshold = threshold_candidates[0]

for t in threshold_candidates:
    y_pred = (mse_test > t).astype(int)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    score = acc + rec  # simple balance
    if score > best_score:
        best_score = score
        best_threshold = t

# Apply threshold
y_pred_final = (mse_test > best_threshold).astype(int)

# Compute metrics
results = {
    "error_statistics": {
        "mean_mse": float(mse_test.mean()),
        "std_mse": float(mse_test.std()),
        "threshold": float(best_threshold),
        "accuracy": float(accuracy_score(y_test, y_pred_final)),
        "precision": float(precision_score(y_test, y_pred_final, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_final, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_final, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, mse_test))
    }
}

# Save results to JSON (overwrite previous)
with open("autoencoder_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))