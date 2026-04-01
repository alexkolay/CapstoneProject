from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_MODEL_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = NEW_MODEL_ROOT / "artifacts"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if np.isclose(lo, hi):
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def rank_average(*arrays: Iterable[np.ndarray]) -> np.ndarray:
    stacked = [np.asarray(arr, dtype=float).ravel() for arr in arrays]
    if not stacked:
        return np.array([], dtype=float)
    return np.mean(np.vstack([normalize_scores(arr) for arr in stacked]), axis=0)


def safe_stratified_split_indices(y: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    y = np.asarray(y)
    stratify = y if len(np.unique(y)) > 1 else None
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train_idx, val_idx
