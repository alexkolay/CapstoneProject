from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .common import PROJECT_ROOT, ensure_dir

DATASET_PATH = PROJECT_ROOT / "spacetrack_starlink.json"
OUTPUT_PATH = ensure_dir(PROJECT_ROOT / "new_model" / "artifacts") / "spacetrack_preprocessed.pkl"

ORBITAL_FEATURES = [
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
]

ALTITUDE_FEATURES = ["APOAPSIS", "PERIAPSIS"]

ANGLE_FEATURES = [
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame[ORBITAL_FEATURES + ALTITUDE_FEATURES] = frame[ORBITAL_FEATURES + ALTITUDE_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )
    frame["orbit_span"] = frame["APOAPSIS"] - frame["PERIAPSIS"]
    frame["orbit_midpoint"] = (frame["APOAPSIS"] + frame["PERIAPSIS"]) / 2.0
    frame["motion_stability"] = frame["MEAN_MOTION_DOT"].abs() / frame["MEAN_MOTION"].replace(0, np.nan)
    frame["curvature_strength"] = frame["MEAN_MOTION_DDOT"].abs()
    frame["eccentricity_energy"] = frame["ECCENTRICITY"] ** 2
    frame["bstar_magnitude"] = frame["BSTAR"].abs()
    for column in ANGLE_FEATURES:
        radians = np.deg2rad(frame[column])
        frame[f"{column.lower()}_sin"] = np.sin(radians)
        frame[f"{column.lower()}_cos"] = np.cos(radians)
    return frame


def main() -> None:
    with open(DATASET_PATH, "r") as handle:
        df = pd.DataFrame(json.load(handle))

    original_count = len(df)
    df = df[df["DECAY_DATE"].isna()].copy()
    active_count = len(df)

    df = add_features(df)
    df = df.dropna(subset=ORBITAL_FEATURES + ALTITUDE_FEATURES)

    feature_cols = [
        *ORBITAL_FEATURES,
        "orbit_span",
        "orbit_midpoint",
        "motion_stability",
        "curvature_strength",
        "eccentricity_energy",
        "bstar_magnitude",
        *[f"{column.lower()}_sin" for column in ANGLE_FEATURES],
        *[f"{column.lower()}_cos" for column in ANGLE_FEATURES],
    ]

    split_index = int(len(df) * 0.8)
    df_train = df.iloc[:split_index].reset_index(drop=True)
    df_test = df.iloc[split_index:].reset_index(drop=True)

    payload = {
        "df_train": df_train,
        "df_test": df_test,
        "feature_cols": feature_cols,
        "dataset_path": str(DATASET_PATH),
        "original_count": original_count,
        "active_count": active_count,
    }

    with open(OUTPUT_PATH, "wb") as handle:
        pickle.dump(payload, handle)

    print(
        f"Saved {OUTPUT_PATH.name}: {original_count} rows -> {active_count} active -> {len(df)} usable; "
        f"train={len(df_train)} test={len(df_test)}"
    )


if __name__ == "__main__":
    main()
