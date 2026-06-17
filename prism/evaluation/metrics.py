"""Evaluation metrics for PRISM modules.

Provides precision / recall / F1 / accuracy for classification tasks (e.g.
hire / no-hire decisions or binarised OCEAN traits) and MAE / RMSE / R2 for the
personality regression head.

Uses scikit-learn when installed; otherwise falls back to dependency-free numpy
implementations so the scripts always run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {"accuracy": self.accuracy, "precision": self.precision,
                "recall": self.recall, "f1": self.f1}


@dataclass
class RegressionReport:
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "macro",
) -> ClassificationReport:
    """Compute accuracy, precision, recall and F1.

    Falls back to a numpy implementation (binary / macro-averaged) when
    scikit-learn is not available.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        return ClassificationReport(
            accuracy=float(accuracy_score(yt, yp)),
            precision=float(precision_score(yt, yp, average=average, zero_division=0)),
            recall=float(recall_score(yt, yp, average=average, zero_division=0)),
            f1=float(f1_score(yt, yp, average=average, zero_division=0)),
        )
    except Exception:
        return _classification_numpy(yt, yp)


def evaluate_regression(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> RegressionReport:
    """Compute MAE, RMSE and R^2 (numpy fallback if sklearn missing)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return RegressionReport(mae=mae, rmse=rmse, r2=r2)


def _classification_numpy(yt: np.ndarray, yp: np.ndarray) -> ClassificationReport:
    """Macro-averaged classification metrics without scikit-learn."""
    labels = np.unique(np.concatenate([yt, yp]))
    accuracy = float(np.mean(yt == yp))
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = float(np.sum((yp == label) & (yt == label)))
        fp = float(np.sum((yp == label) & (yt != label)))
        fn = float(np.sum((yp != label) & (yt == label)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return ClassificationReport(
        accuracy=accuracy,
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        f1=float(np.mean(f1s)),
    )
