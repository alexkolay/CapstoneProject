"""
This file is for testing telemetry anomaly detection models based on OPTSAT_dataset.csv and return different evaluation metrics, such as per-channel precision/recall/F1,
model comparison summary, and Random Forest feature importances.

This is a list of the models trained:
    1. Logistic Regression 
    2. Linear SVM           
    3. Random Forest         
    4. Gradient Boosting    

To run this file, first run preprocess_OPTSAT.py to generate optsat_preprocessed.pkl. Then, this file createa a file called optsat_test_predictions.csv
that contains held-out test rows only, best model by test F1, columns segment, channel, y_actual (whether anomaly or not), y_pred (prediction on anomaly or not anomaly),
and correct (whether predicted = actual).
"""

import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# ── Load preprocessed data ────────────────────────────────────────────────────
with open("optsat_preprocessed.pkl", "rb") as f:
    data = pickle.load(f)

X_train       = data["X_train"]
X_test        = data["X_test"]
y_train       = data["y_train"]
y_test        = data["y_test"]
feature_names = data["feature_names"]
channel_test  = data["channel_test"]
segment_test  = data.get("segment_test")
sampling_test = data.get("sampling_test")

num_channels = sum(1 for feature in feature_names if feature.startswith("ch_"))
print(f"Telemetry: train={len(y_train)}  test={len(y_test)}  "
      f"| anomaly% train={y_train.mean():.1%} test={y_test.mean():.1%}  "
      f"| {len(feature_names)} features ({num_channels} channel dummies)")

def evaluation_per_channel(y_true, y_pred, channels):
    """Print F1, precision, and recall scores by channel."""
    print(f"  {'Channel':<12} {'N':>5} {'Anom':>5} {'Pred':>5} "
          f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*52}")
    for channel in sorted(set(channels)):
        mask  = channels == channel
        yt    = y_true[mask]
        yp    = y_pred[mask]
        n     = len(yt)
        anom  = yt.sum()
        pred  = yp.sum()
        prec  = precision_score(yt, yp, zero_division=0)
        rec   = recall_score(yt, yp, zero_division=0)
        f1    = f1_score(yt, yp, zero_division=0)
        print(f"  {channel:<12} {n:>5} {int(anom):>5} {int(pred):>5} "
              f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")

def bootstrap_cv(model, X, y, n_reps=5, random_state=0, fit_sample_weight=None):
    """
    Perform bootstrap resampling to estimate cross-validated F1 score. We use this method to preserve the class imbalance and channel distribution in each resample.
    """
    rng = np.random.default_rng(random_state)
    scores = []
    n = len(y)
    for i in range(n_reps):
        idx = rng.integers(0, n, size=n)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) == 0:
            continue
        kw = {}
        if fit_sample_weight is not None:
            kw["sample_weight"] = fit_sample_weight[idx]
        model.fit(X[idx], y[idx], **kw)
        pred = model.predict(X[oob])
        scores.append(f1_score(y[oob], pred, average="weighted", zero_division=0))
    return float(np.mean(scores))

def evaluate(name, model, X_tr, y_tr, X_te, y_te,
             channels_te, needs_calibration=False, fit_sample_weight=None):

    cv_f1 = bootstrap_cv(model, X_tr, y_tr, fit_sample_weight=fit_sample_weight)

    if needs_calibration:
        cal = CalibratedClassifierCV(model, cv=5)
        cal.fit(X_tr, y_tr)
        y_prob = cal.predict_proba(X_te)[:, 1]
        y_pred = cal.predict(X_te)
    else:
        fit_kw = {}
        if fit_sample_weight is not None:
            fit_kw["sample_weight"] = fit_sample_weight
        model.fit(X_tr, y_tr, **fit_kw)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_te, y_prob) if y_prob is not None else float("nan")

    print(f"\n{name}")
    print(f"  Acc={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  ROC-AUC={auc:.4f}")
    print("  Per-channel (test):")
    evaluation_per_channel(y_te, y_pred, channels_te)

    return {"model": name, "cv_f1": cv_f1, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1, "auc": auc,
            "y_pred": y_pred, "y_prob": y_prob}

lr = LogisticRegression(
    multi_class="ovr", solver="saga", class_weight="balanced",
    max_iter=1000, C=1.0, n_jobs=-1, random_state=42,
)
r1 = evaluate("Logistic Regression (OvR, saga, C=1.0)",
              lr, X_train, y_train, X_test, y_test, channel_test)

svm = LinearSVC(
    multi_class="ovr", class_weight="balanced",
    dual=False, max_iter=10000, C=1.0, random_state=42,
)
r2 = evaluate("Linear SVM (OvR, LinearSVC, C=1.0)",
              svm, X_train, y_train, X_test, y_test, channel_test,
              needs_calibration=True)

rf = RandomForestClassifier(
    n_estimators=100, criterion="gini",
    class_weight="balanced_subsample", n_jobs=-1, random_state=42,
)
r3 = evaluate("Random Forest (100 trees, Gini, balanced_subsample)",
              rf, X_train, y_train, X_test, y_test, channel_test)

sw_train = compute_sample_weight("balanced", y_train)
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42,
)
r4 = evaluate("Gradient Boosting (200 trees, depth=4, lr=0.1)",
              gb, X_train, y_train, X_test, y_test, channel_test,
              fit_sample_weight=sw_train)

results = sorted([r1, r2, r3, r4], key=lambda x: x["f1"], reverse=True)
print("\nSummary (sorted by test F1)")
print(f"{'Model':<44} {'CV-F1':>7} {'Acc':>7} {'F1':>7} {'AUC':>7}")
for r in results:
    print(f"{r['model'][:44]:<44} {r['cv_f1']:>7.4f} {r['accuracy']:>7.4f} "
          f"{r['f1']:>7.4f} {r['auc']:>7.4f}")

print("\nTop 15 features (Random Forest importance)")
importances = rf.feature_importances_
ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(ranked[:15], 1):
    print(f"  {i:>2}. {feat:<30} {imp:.4f}")

best = results[0]
print(f"\nBest by test F1: {best['model']}")
model_map = {
    r1["model"]: lr,
    r2["model"]: svm,
    r3["model"]: rf,
    r4["model"]: gb,
}
best_key = next(k for k in model_map if k.startswith(best["model"].split()[0]))
with open("best_optsat_model.pkl", "wb") as f:
    pickle.dump({"model": model_map[best_key], "scaler": data["scaler"],
                 "feature_names": feature_names,
                 "channel_cols": data["channel_cols"]}, f)
print("Saved → best_optsat_model.pkl")

n_te = len(y_test)
rows = {
    "channel": channel_test,
    "y_actual": y_test,
    "y_pred": best["y_pred"],
}
if segment_test is not None and len(segment_test) == n_te:
    rows["segment"] = segment_test
if sampling_test is not None and len(sampling_test) == n_te:
    rows["sampling"] = sampling_test
df_pred = pd.DataFrame(rows)
df_pred["correct"] = (df_pred["y_actual"] == df_pred["y_pred"]).astype(int)
cols = [c for c in ("segment", "channel", "sampling") if c in df_pred.columns]
cols += ["y_actual", "y_pred", "correct"]
df_pred = df_pred[cols]
df_pred.to_csv("optsat_test_predictions.csv", index=False)
print("Saved → optsat_test_predictions.csv  (test split; join segment to OPTSAT_dataset.csv)")
