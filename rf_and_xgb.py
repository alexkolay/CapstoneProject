"""
Telemetry anomaly classification pipeline
First draft for OPSSAT-AD style tabular telemetry features.

Expected columns
----------------
Target:
- anomaly

Split indicator:
- train   (1=train, 0=test)

Categorical:
- channel

Identifier to drop:
- segment

Typical usage
-------------
python telemetry_anomaly_model.py --csv dataset.csv

Notes
-----
- This version starts with Random Forest and XGBoost.
- If xgboost is not installed, the script will still run the Random Forest pipeline.
- One-hot encoding is applied to `channel`.
- The provided `train` column is used for the first split.
- Cross-validation is included on the training subset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float | None
    confusion: List[List[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telemetry anomaly classification pipeline")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_outputs",
        help="Directory to write reports and artifacts",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.80,
        help="Preferred minimum recall when selecting a probability threshold",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds on the training subset",
    )
    return parser.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def basic_validation(df: pd.DataFrame) -> None:
    required = {"anomaly", "train", "channel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["anomaly"].isna().any():
        raise ValueError("Target column `anomaly` contains missing values.")

    if not set(df["anomaly"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target column `anomaly` must be binary (0/1).")

    if not set(df["train"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Split column `train` must be binary (0/1).")


def summarize_data(df: pd.DataFrame) -> Dict:
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "anomaly_rate_overall": float(df["anomaly"].mean()),
        "train_rows": int((df["train"] == 1).sum()),
        "test_rows": int((df["train"] == 0).sum()),
        "train_anomaly_rate": float(df.loc[df["train"] == 1, "anomaly"].mean()),
        "test_anomaly_rate": float(df.loc[df["train"] == 0, "anomaly"].mean()),
        "unique_channels": int(df["channel"].nunique(dropna=True)),
        "channels": sorted(df["channel"].dropna().astype(str).unique().tolist()),
        "missing_by_column": df.isna().sum().to_dict(),
    }
    return summary


def choose_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    drop_cols = ["anomaly", "train"]
    if "segment" in df.columns:
        drop_cols.append("segment")

    feature_cols = [c for c in df.columns if c not in drop_cols]

    categorical_cols = []
    if "channel" in feature_cols:
        categorical_cols.append("channel")

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return feature_cols, numeric_cols, categorical_cols


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["train"] == 1].copy()
    test_df = df[df["train"] == 0].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Using the `train` column produced an empty train or test set.")

    return train_df, test_df


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def make_random_forest_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def make_xgboost_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale_pos_weight: float,
) -> Pipeline:
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed in this environment.")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )
    return {metric: float(np.mean(values)) for metric, values in scores.items() if metric.startswith("test_")}


def choose_threshold_for_target_recall(
    y_true: pd.Series,
    y_proba: np.ndarray,
    target_recall: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve returns recall/precision arrays of len(thresholds)+1
    # Align metrics to thresholds by dropping the final element.
    if len(thresholds) == 0:
        return 0.5

    candidate_rows = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        candidate_rows.append((float(t), float(p), float(r)))

    qualifying = [row for row in candidate_rows if row[2] >= target_recall]
    if qualifying:
        # among thresholds meeting recall target, choose best precision
        best = max(qualifying, key=lambda x: (x[1], x[0]))
        return best[0]

    # fallback: choose threshold with best F1
    best_t = 0.5
    best_f1 = -1.0
    for t, p, r in candidate_rows:
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_recall: float,
) -> ModelResult:
    pipeline.fit(X_train, y_train)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(f"{name} pipeline does not support predict_proba.")

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    threshold = choose_threshold_for_target_recall(y_test, test_proba, target_recall)
    test_pred = (test_proba >= threshold).astype(int)

    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, test_proba)

    try:
        roc_auc = roc_auc_score(y_test, test_proba)
    except ValueError:
        roc_auc = None

    confusion = confusion_matrix(y_test, test_pred).tolist()

    return ModelResult(
        name=name,
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        pr_auc=float(pr_auc),
        roc_auc=None if roc_auc is None else float(roc_auc),
        confusion=confusion,
    )


def extract_feature_importance(
    pipeline: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get transformed feature names after one-hot encoding.
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df_imp = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        return df_imp

    return pd.DataFrame(columns=["feature", "importance"])


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    basic_validation(df)

    summary = summarize_data(df)
    save_json(output_dir / "data_summary.json", summary)

    train_df, test_df = split_train_test(df)

    feature_cols, numeric_cols, categorical_cols = choose_feature_columns(df)

    X_train = train_df[feature_cols]
    y_train = train_df["anomaly"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["anomaly"].astype(int)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    models: List[Tuple[str, Pipeline]] = [
        ("random_forest", make_random_forest_pipeline(numeric_cols, categorical_cols))
    ]

    if HAS_XGBOOST:
        models.append(
            (
                "xgboost",
                make_xgboost_pipeline(
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    scale_pos_weight=scale_pos_weight,
                ),
            )
        )

    results: List[Dict] = []
    report_lines: List[str] = []

    report_lines.append("Telemetry Anomaly Classification Report")
    report_lines.append("=" * 42)
    report_lines.append("")
    report_lines.append("Data summary:")
    for key, value in summary.items():
        report_lines.append(f"- {key}: {value}")
    report_lines.append("")
    report_lines.append(f"Using features: {feature_cols}")
    report_lines.append(f"Numeric columns: {numeric_cols}")
    report_lines.append(f"Categorical columns: {categorical_cols}")
    report_lines.append(f"scale_pos_weight (for XGBoost): {scale_pos_weight:.4f}")
    report_lines.append("")

    for model_name, pipeline in models:
        cv_scores = cross_validate_model(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            cv_folds=args.cv_folds,
        )
        result = evaluate_model(
            name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            target_recall=args.target_recall,
        )

        results.append(
            {
                "model": result.name,
                "threshold": result.threshold,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "pr_auc": result.pr_auc,
                "roc_auc": result.roc_auc,
                "confusion_matrix": result.confusion,
                "cv_scores": cv_scores,
            }
        )

        report_lines.append(f"Model: {result.name}")
        report_lines.append("-" * (7 + len(result.name)))
        report_lines.append(f"Threshold: {result.threshold:.4f}")
        report_lines.append(f"Test precision: {result.precision:.4f}")
        report_lines.append(f"Test recall: {result.recall:.4f}")
        report_lines.append(f"Test F1: {result.f1:.4f}")
        report_lines.append(f"Test PR-AUC: {result.pr_auc:.4f}")
        report_lines.append(f"Test ROC-AUC: {result.roc_auc}")
        report_lines.append(f"Confusion matrix: {result.confusion}")
        report_lines.append("CV means:")
        for metric, score in cv_scores.items():
            report_lines.append(f"  - {metric}: {score:.4f}")
        report_lines.append("")

        # Save feature importances after fitting.
        feature_importance = extract_feature_importance(
            pipeline=pipeline,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        if not feature_importance.empty:
            feature_importance.to_csv(
                output_dir / f"{model_name}_feature_importance.csv",
                index=False,
            )

        # Also save a full classification report using the chosen threshold.
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= result.threshold).astype(int)
        class_report = classification_report(y_test, test_pred, digits=4)
        save_text(output_dir / f"{model_name}_classification_report.txt", class_report)

    save_json(output_dir / "model_results.json", {"results": results})
    save_text(output_dir / "report.txt", "\n".join(report_lines))

    print("Done.")
    print(f"Artifacts written to: {output_dir.resolve()}")
    print("Primary files:")
    print(f"- {output_dir / 'data_summary.json'}")
    print(f"- {output_dir / 'model_results.json'}")
    print(f"- {output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
