import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.legend_handler import HandlerPathCollection

# The LOF model works well for celestrak_starlink.csv because it compares each satellite
# to its local neighborhood and identifies satellites whose orbital behavior is unusual
# relative to nearby satellites. This is useful for Starlink data because most satellites
# should have similar orbital characteristics, while unusual ones may indicate anomalies.

# run by python3 local_outlier_factor/local_outlier_factor_celestrak.py

df = pd.read_csv("celestrak_starlink.csv")

# choose main numeric features
features = [
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT"
]

# convert selected columns to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# keep only rows with all selected features present
model_df = df.dropna(subset=features).copy()

# scale features because LOF is distance-based
scaler = StandardScaler()
X = scaler.fit_transform(model_df[features])

# fit LOF model
# contamination is the expected fraction of anomalies
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)

# LOF predictions: -1 = anomaly, 1 = normal
y_pred = clf.fit_predict(X)

# LOF scores: more negative means more abnormal
lof_scores = clf.negative_outlier_factor_

# store results
model_df["predicted_anomaly"] = (y_pred == -1).astype(int)
model_df["lof_score"] = lof_scores

# keep only predicted anomalies
anomalies_df = model_df[model_df["predicted_anomaly"] == 1].copy()

# sort strongest anomalies first
anomalies_df = anomalies_df.sort_values(by="lof_score")

# columns to save
output_columns = [
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "predicted_anomaly",
    "lof_score",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT"
]

anomalies_df = anomalies_df[output_columns]

# save output
anomalies_df.to_csv("local_outlier_factor/anomalies_lof_celestrak.csv", index=False)

print("Done.")
print("Anomalies saved to local_outlier_factor/anomalies_lof_celestrak.csv")
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

plt.scatter(X_2d[:, 0], X_2d[:, 1], s=20, label="Satellites")

scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    s=1200 * radius,
    edgecolors="r",
    facecolors="none",
    label="LOF outlier score"
)

# highlight predicted anomalies
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
plt.title("Local Outlier Factor (LOF) - CelesTrak Starlink")
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)

plt.tight_layout()
plt.savefig("local_outlier_factor/lof_celestrak_plot.png", dpi=300)
plt.show()