import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib.legend_handler import HandlerPathCollection

# run by python3 local_outlier_factor/local_outlier_factor_dataset.py
# accuracy is printed on top of the console, NOT saved to the output csv file
# Accuracy: 79.60%

df = pd.read_csv("dataset.csv")

features = [
    "mean",
    "std",
    "var",
    "skew",
    "kurtosis",
    "diff_var",
    "diff2_var",
    "n_peaks"
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

model_df = df.dropna(subset=features).copy()

scaler = StandardScaler()
X = scaler.fit_transform(model_df[features])

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.20)

y_pred = clf.fit_predict(X)

lof_scores = clf.negative_outlier_factor_

model_df["predicted_anomaly"] = (y_pred == -1).astype(int)
model_df["lof_score"] = lof_scores

accuracy = accuracy_score(model_df["anomaly"], model_df["predicted_anomaly"]) * 100
print(f"Accuracy: {accuracy:.2f}%")

anomalies_df = model_df[model_df["predicted_anomaly"] == 1].copy()

anomalies_df = anomalies_df.sort_values(by="lof_score")

output_columns = [
    "segment",
    "anomaly",
    "predicted_anomaly",
    "lof_score"
] + features

anomalies_df = anomalies_df[output_columns]

anomalies_df.to_csv("local_outlier_factor/anomalies_lof_dataset.csv", index=False)

print("Done.")
print("Anomalies saved to local_outlier_factor/anomalies_lof_dataset.csv")
print("Number of predicted anomalies:", len(anomalies_df))
print(anomalies_df.head(20).to_string(index=False))

# ----------------------------
# Graph section
# ----------------------------

# reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# convert LOF scores into circle sizes
# more negative = more abnormal, so bigger circle
radius = (lof_scores.max() - lof_scores) / (lof_scores.max() - lof_scores.min() + 1e-10)

def update_legend_marker_size(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([50])

plt.figure(figsize=(10, 7))

plt.scatter(X_2d[:, 0], X_2d[:, 1], s=20, label="Segments")

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    s=1200 * radius,
    edgecolors="r",
    facecolors="none",
    label="LOF outlier score"
)

anomaly_mask = model_df["predicted_anomaly"] == 1
plt.scatter(
    X_2d[anomaly_mask, 0],
    X_2d[anomaly_mask, 1],
    s=40,
    marker="x",
    label="Predicted anomalies"
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Local Outlier Factor (LOF) - Dataset")
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)

plt.tight_layout()
plt.savefig("local_outlier_factor/lof_dataset_plot.png", dpi=300)
plt.show()