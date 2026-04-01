"""
Satellite Anomaly Detection — Gradient Boosting Template
=========================================================
Dataset columns assumed:
  segment, anomaly, train, channel, sampling,
  duration, len, mean, var, std, kurtosis, skew,
  n_peaks, smooth10_n_peaks, smooth20_n_peaks,
  diff_peaks, diff2_peaks, diff_var, diff2_var,
  gaps_squared, len_weighted, var_div_duration, var_div_len
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ── 1. Load data ───────────────────────────────────────────────────────────────

df = pd.read_csv("zenodo.csv")   # ← replace with your file path

# ── 2. Feature / target / split setup ─────────────────────────────────────────

# Encode the channel string column as an integer
le = LabelEncoder()
df["channel_enc"] = le.fit_transform(df["channel"])

# Drop ALL non-feature cols including the original string channel
DROP_COLS = ["segment", "anomaly", "train", "channel", "sampling"]

FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]
TARGET_COL   = "anomaly"

# Use the pre-existing train flag for splitting
train_df = df[df["train"] == 1].copy()
test_df  = df[df["train"] == 0].copy()

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET_COL]

print(f"Train size : {len(X_train)}  |  anomaly rate: {y_train.mean():.2%}")
print(f"Test  size : {len(X_test)}   |  anomaly rate: {y_test.mean():.2%}")

# ── 3. Handle class imbalance ──────────────────────────────────────────────────

# scale_pos_weight = #negative / #positive
# Tells XGBoost to up-weight the minority class automatically.
neg  = (y_train == 0).sum()
pos  = (y_train == 1).sum()
spw  = neg / pos
print(f"\nscale_pos_weight = {spw:.2f}  ({neg} normal : {pos} anomaly)")

# ── 4. Model definition ────────────────────────────────────────────────────────

model = xgb.XGBClassifier(
    # --- core boosting params ---
    n_estimators      = 300,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,

    # --- class imbalance ---
    scale_pos_weight  = spw,

    # --- reproducibility ---
    random_state      = 42,

    # --- speed ---
    n_jobs            = -1,
    tree_method       = "hist",     # fast histogram method

    # --- evaluation during training ---
    eval_metric       = ["logloss", "auc"],
    early_stopping_rounds = 20,     # stop if no improvement for 20 rounds
)

# ── 5. Train ───────────────────────────────────────────────────────────────────

model.fit(
    X_train, y_train,
    eval_set     = [(X_train, y_train), (X_test, y_test)],
    verbose      = 50,   # print every 50 rounds
)

print(f"\nBest iteration: {model.best_iteration}")

# ── 6. Evaluate ────────────────────────────────────────────────────────────────

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("\n── Classification report ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

roc_auc = roc_auc_score(y_test, y_pred_prob)
pr_auc  = average_precision_score(y_test, y_pred_prob)
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}   ← most informative for imbalanced data")

# ── 7. Confusion matrix ────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Normal", "Anomaly"],
    cmap="Blues",
    ax=ax,
)
ax.set_title("Confusion matrix — test set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── 8. Feature importance ──────────────────────────────────────────────────────

importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
top_n      = importance.nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(7, 5))
top_n.plot(kind="barh", ax=ax, color="#1D9E75")
ax.set_title("Top 15 features by importance (gain)")
ax.set_xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nTop 5 features:")
print(importance.nlargest(5).to_string())

# ── 9. Threshold tuning (optional) ────────────────────────────────────────────
# Default threshold is 0.5; lowering it catches more anomalies (higher recall)
# at the cost of more false positives. Uncomment to experiment.

# from sklearn.metrics import precision_recall_curve
# precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
# # Pick threshold that maximises F1
# f1_scores    = 2 * precision * recall / (precision + recall + 1e-8)
# best_thresh  = thresholds[f1_scores[:-1].argmax()]
# print(f"\nOptimal threshold (max F1): {best_thresh:.3f}")
# y_pred_tuned = (y_pred_prob >= best_thresh).astype(int)
# print(classification_report(y_test, y_pred_tuned, target_names=["Normal", "Anomaly"]))

# ── 10. Save model ─────────────────────────────────────────────────────────────

model.save_model("satellite_anomaly_xgb.json")
print("\nModel saved to satellite_anomaly_xgb.json")

# To reload later:
# loaded_model = xgb.XGBClassifier()
# loaded_model.load_model("satellite_anomaly_xgb.json")