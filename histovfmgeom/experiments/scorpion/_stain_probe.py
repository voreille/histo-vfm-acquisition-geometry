from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StainProbeData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_train_sources: int
    n_test_sources: int


def _subsample(
    x: np.ndarray,
    y: np.ndarray,
    max_examples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_examples is None or len(x) <= max_examples:
        return x, y
    rng = np.random.default_rng(seed)
    labels = np.asarray(y)
    classes = np.unique(labels)
    per_class = max(1, max_examples // max(1, len(classes)))
    parts: list[np.ndarray] = []
    for cls in classes:
        idx = np.nonzero(labels == cls)[0]
        parts.append(idx if len(idx) <= per_class else rng.choice(idx, per_class, replace=False))
    selected = np.unique(np.concatenate(parts))
    if len(selected) < max_examples:
        rest = np.setdiff1d(np.arange(len(labels)), selected, assume_unique=False)
        n_extra = min(max_examples - len(selected), len(rest))
        if n_extra > 0:
            selected = np.concatenate([selected, rng.choice(rest, n_extra, replace=False)])
    if len(selected) > max_examples:
        selected = rng.choice(selected, max_examples, replace=False)
    selected = np.sort(selected.astype(np.int64))
    return x[selected], labels[selected]


def build_stain_probe(
    *,
    stain_features: np.ndarray,
    stain_metadata: pd.DataFrame,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    source_row_index_col: str = "source_row_index",
    label_col: str = "target_id",
    max_examples_per_split: int | None = None,
    seed: int = 0,
) -> StainProbeData | None:
    if len(train_rows) == 0 or len(test_rows) == 0:
        logger.warning("Skipping stain probe: empty split (n_train=%d, n_test=%d).", len(train_rows), len(test_rows))
        return None

    x_train = stain_features[train_rows].astype(np.float32, copy=False)
    x_test  = stain_features[test_rows].astype(np.float32, copy=False)
    y_train = stain_metadata.iloc[train_rows][label_col].astype(str).to_numpy()
    y_test  = stain_metadata.iloc[test_rows][label_col].astype(str).to_numpy()

    x_train, y_train = _subsample(x_train, y_train, max_examples_per_split, seed)
    x_test,  y_test  = _subsample(x_test,  y_test,  max_examples_per_split, seed + 10_000)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        logger.warning("Skipping stain probe: fewer than two labels in train or test.")
        return None

    return StainProbeData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        n_train_sources=int(stain_metadata.iloc[train_rows][source_row_index_col].nunique()),
        n_test_sources=int(stain_metadata.iloc[test_rows][source_row_index_col].nunique()),
    )
