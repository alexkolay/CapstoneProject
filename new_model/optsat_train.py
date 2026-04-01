from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.inspection import permutation_importance

from .common import ARTIFACT_ROOT, ensure_dir, save_json, safe_stratified_split_indices

PREPROCESSED_PATH = ARTIFACT_ROOT / "optsat_preprocessed.pkl"
MODEL_DIR = ensure_dir(ARTIFACT_ROOT / "optsat")


def load_payload() -> dict:
    with open(PREPROCESSED_PATH, "rb") as handle:
        return pickle.load(handle)


def build_pipeline(feature_frame: pd.DataFrame) -> Pipeline:
    numeric_cols = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = feature_frame.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    classifier = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
    }
    return metrics


def main() -> None:
    payload = load_payload()
    train_df = payload["train_df"].copy()
    test_df = payload["test_df"].copy()
    feature_cols = payload["feature_cols"]
    target_col = payload["target_col"]

    x_train_full = train_df[feature_cols]
    y_train_full = train_df[target_col].astype(int).to_numpy()
    x_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int).to_numpy()

    train_idx, val_idx = safe_stratified_split_indices(y_train_full, test_size=0.2, random_state=42)
    x_train = x_train_full.iloc[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full.iloc[val_idx]
    y_val = y_train_full[val_idx]

    pipeline = build_pipeline(x_train)
    pipeline.fit(x_train, y_train)

    val_prob = pipeline.predict_proba(x_val)[:, 1]
    threshold, val_best_f1 = choose_threshold(y_val, val_prob)

    test_prob = pipeline.predict_proba(x_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(int)

    model_path = MODEL_DIR / "optsat_extratrees.joblib"
    joblib.dump(pipeline, model_path)

    predictions = test_df[["segment", "channel"]].copy()
    predictions["y_actual"] = y_test
    predictions["y_prob"] = test_prob
    predictions["y_pred"] = test_pred
    predictions["correct"] = (predictions["y_actual"] == predictions["y_pred"]).astype(int)
    predictions_path = MODEL_DIR / "optsat_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    fitted_preprocessor = pipeline.named_steps["preprocess"]
    feature_names = fitted_preprocessor.get_feature_names_out().tolist()
    importances = pipeline.named_steps["model"].feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)

    save_json(
        MODEL_DIR / "optsat_metrics.json",
        {
            "validation_best_f1": val_best_f1,
            "selected_threshold": threshold,
            "test_metrics": test_metrics,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(feature_cols)),
            "top_features": [{"feature": name, "importance": float(score)} for name, score in ranked[:20]],
        },
    )

    print(f"Saved model to {model_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Selected threshold: {threshold:.2f}")
    print(
        f"Test F1={test_metrics['f1']:.4f}  Accuracy={test_metrics['accuracy']:.4f}  "
        f"Precision={test_metrics['precision']:.4f}  Recall={test_metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
