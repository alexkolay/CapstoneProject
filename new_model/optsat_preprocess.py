from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .common import PROJECT_ROOT, ensure_dir

DATASET_PATH = PROJECT_ROOT / "dataset.csv"
OUTPUT_PATH = ensure_dir(PROJECT_ROOT / "new_model" / "artifacts") / "optsat_preprocessed.pkl"

DROP_COLUMNS = {"segment", "anomaly", "train"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["peak_density"] = frame["n_peaks"] / frame["len"].replace(0, np.nan)
    frame["smoothed_peak_ratio"] = frame["smooth10_n_peaks"] / frame["smooth20_n_peaks"].replace(0, np.nan)
    frame["diff_peak_gap"] = frame["diff_peaks"] - frame["diff2_peaks"]
    frame["variance_per_duration"] = frame["var"] / frame["duration"].replace(0, np.nan)
    frame["variance_per_length"] = frame["var"] / frame["len"].replace(0, np.nan)
    frame["absolute_mean"] = frame["mean"].abs()
    frame["kurtosis_abs"] = frame["kurtosis"].abs()
    frame["skew_abs"] = frame["skew"].abs()
    frame["channel"] = frame["channel"].astype("category")
    return frame


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df = build_features(df)

    required = {"anomaly", "train", "channel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_df = df[df["train"] == 1].copy()
    test_df = df[df["train"] == 0].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Expected both train and test rows based on the train column.")

    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    payload = {
        "train_df": train_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "feature_cols": feature_cols,
        "target_col": "anomaly",
        "split_col": "train",
        "dataset_path": str(DATASET_PATH),
    }

    with open(OUTPUT_PATH, "wb") as handle:
        pickle.dump(payload, handle)

    print(f"Saved {OUTPUT_PATH.name} with {len(train_df)} train rows and {len(test_df)} test rows.")


if __name__ == "__main__":
    main()
