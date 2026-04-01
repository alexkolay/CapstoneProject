from __future__ import annotations

import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .common import ARTIFACT_ROOT, ensure_dir, normalize_scores, rank_average, save_json

PREPROCESSED_PATH = ARTIFACT_ROOT / "spacetrack_preprocessed.pkl"
MODEL_DIR = ensure_dir(ARTIFACT_ROOT / "spacetrack")


def load_payload() -> dict:
    with open(PREPROCESSED_PATH, "rb") as handle:
        return pickle.load(handle)


def build_numeric_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return frame[feature_cols].copy()


def main() -> None:
    payload = load_payload()
    df_train = payload["df_train"].copy()
    df_test = payload["df_test"].copy()
    feature_cols = payload["feature_cols"]

    x_train = build_numeric_frame(df_train, feature_cols)
    x_test = build_numeric_frame(df_test, feature_cols)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    x_train_scaled = numeric_pipe.fit_transform(x_train)
    x_test_scaled = numeric_pipe.transform(x_test)

    if hasattr(x_train_scaled, "toarray"):
        x_train_scaled = x_train_scaled.toarray()
    if hasattr(x_test_scaled, "toarray"):
        x_test_scaled = x_test_scaled.toarray()

    iso = IsolationForest(
        n_estimators=400,
        contamination=0.04,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(x_train_scaled)
    iso_scores = normalize_scores(-iso.decision_function(x_test_scaled))

    pca = PCA(n_components=0.95, random_state=42)
    pca.fit(x_train_scaled)
    reconstructed = pca.inverse_transform(pca.transform(x_test_scaled))
    pca_scores = normalize_scores(np.sqrt(np.mean((x_test_scaled - reconstructed) ** 2, axis=1)))

    combined_scores = rank_average(iso_scores, pca_scores)
    anomaly_threshold = float(np.quantile(combined_scores, 0.96))
    anomaly_flag = (combined_scores >= anomaly_threshold).astype(int)

    model_path = MODEL_DIR / "spacetrack_isoforest_pca.joblib"
    joblib.dump({"scaler": numeric_pipe, "isolation_forest": iso, "pca": pca, "feature_cols": feature_cols}, model_path)

    results = df_test[["OBJECT_NAME", "NORAD_CAT_ID", "INCLINATION", "APOAPSIS", "PERIAPSIS", "BSTAR"]].copy()
    results["iso_score"] = iso_scores
    results["pca_score"] = pca_scores
    results["combined_score"] = combined_scores
    results["anomaly_flag"] = anomaly_flag

    results_path = MODEL_DIR / "spacetrack_anomaly_scores.csv"
    results.to_csv(results_path, index=False)

    reconstruction_rmse = float(np.sqrt(np.mean((x_test_scaled - reconstructed) ** 2)))
    save_json(
        MODEL_DIR / "spacetrack_metrics.json",
        {
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "feature_count": int(len(feature_cols)),
            "combined_threshold": anomaly_threshold,
            "flagged_count": int(anomaly_flag.sum()),
            "flagged_rate": float(anomaly_flag.mean()),
            "reconstruction_rmse": reconstruction_rmse,
            "top_flagged": results.nlargest(15, "combined_score")[
                ["OBJECT_NAME", "NORAD_CAT_ID", "combined_score"]
            ].to_dict(orient="records"),
        },
    )

    print(f"Saved model to {model_path}")
    print(f"Saved anomaly scores to {results_path}")
    print(f"Combined threshold: {anomaly_threshold:.4f}")
    print(f"Flagged {int(anomaly_flag.sum())} of {len(anomaly_flag)} held-out satellites")


if __name__ == "__main__":
    main()
