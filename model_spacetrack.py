"""
SpaceTrack orbital anomaly detection (active satellites only).

Since all training satellites are active (no decay labels), anomaly detection
is unsupervised. Two models are compared:

    1. Isolation Forest  — random partitioning; anomalies isolate quickly
    2. One-Class SVM     — learns a tight boundary around normal orbital space

Evaluation:
    - Anomaly score distributions on the held-out test set
    - Top flagged satellites with orbital context (periapsis, BSTAR, etc.)
    - Agreement matrix: how often do models agree on the same satellite?

Run preprocess_spacetrack.py first to generate spacetrack_preprocessed.pkl.
"""

import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

# ── Load preprocessed data ────────────────────────────────────────────────────
with open("spacetrack_preprocessed.pkl", "rb") as f:
    data = pickle.load(f)

X_train       = data["X_train"]
X_test        = data["X_test"]
feature_names = data["feature_names"]
df_train      = data["df_train"]
df_test       = data["df_test"]
scaler        = data["scaler"]

CONTAMINATION = 0.02

# ── Helper: normalise raw decision scores to [0, 1] anomaly probability ───────
def normalise_scores(raw_scores):
    """Higher raw score = more anomalous. Normalise to [0,1]."""
    scl = MinMaxScaler()
    return scl.fit_transform(raw_scores.reshape(-1, 1)).ravel()

# ── 1. Isolation Forest ───────────────────────────────────────────────────────
iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION, random_state=42, n_jobs=-1)
iso.fit(X_train)
iso_raw    = iso.predict(X_test)
iso_scores = normalise_scores(-iso.decision_function(X_test))
iso_flags  = (iso_raw == -1).astype(int)

# ── 2. One-Class SVM ──────────────────────────────────────────────────────────
ocsvm = OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale")
ocsvm.fit(X_train)
ocsvm_raw    = ocsvm.predict(X_test)
ocsvm_scores = normalise_scores(-ocsvm.decision_function(X_test))
ocsvm_flags  = (ocsvm_raw == -1).astype(int)

# ── Ensemble ──────────────────────────────────────────────────────────────────
ensemble_scores = (iso_scores + ocsvm_scores) / 2.0
ensemble_flags  = iso_flags.astype(int) + ocsvm_flags.astype(int)
majority_flag   = (ensemble_flags >= 2).astype(int)

df_results = df_test[["OBJECT_NAME", "NORAD_CAT_ID",
                        "INCLINATION", "PERIAPSIS", "APOAPSIS",
                        "MEAN_MOTION", "BSTAR", "MEAN_MOTION_DOT"]].copy()
df_results["iso_score"]      = iso_scores.round(4)
df_results["ocsvm_score"]    = ocsvm_scores.round(4)
df_results["ensemble_score"] = ensemble_scores.round(4)
df_results["models_flagged"] = ensemble_flags
df_results["anomaly_flag"]   = majority_flag

# ── Output ────────────────────────────────────────────────────────────────────
print(f"SpaceTrack: train={len(X_train):,}  test={len(X_test):,}  |  {len(feature_names)} features")
print(f"Flagged (Isolation Forest / One-Class SVM / both): "
      f"{iso_flags.sum()} ({iso_flags.mean():.1%}) / "
      f"{ocsvm_flags.sum()} ({ocsvm_flags.mean():.1%}) / "
      f"{majority_flag.sum()} ({majority_flag.mean():.1%})")

bins   = [-0.001, 0.3, 0.5, 0.7, 1.001]
labels = ["Normal (<0.3)", "Mild (0.3–0.5)", "Moderate (0.5–0.7)", "Severe (>0.7)"]
tiers  = pd.cut(ensemble_scores, bins=bins, labels=labels).value_counts().sort_index()
print("Ensemble score bins:")
for tier, count in tiers.items():
    print(f"  {tier}: {count} ({count/len(ensemble_scores):.1%})")

DISP = ["OBJECT_NAME", "NORAD_CAT_ID", "INCLINATION",
        "PERIAPSIS", "BSTAR", "ensemble_score", "models_flagged"]
print("\nTop 15 by ensemble score:")
print(df_results.nlargest(15, "ensemble_score")[DISP].to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
with open("best_spacetrack_model.pkl", "wb") as f:
    pickle.dump({"model": iso, "scaler": scaler,
                 "feature_names": feature_names, "contamination": CONTAMINATION}, f)
df_results.to_csv("spacetrack_anomaly_scores.csv", index=False)
print("\nSaved: best_spacetrack_model.pkl, spacetrack_anomaly_scores.csv")
