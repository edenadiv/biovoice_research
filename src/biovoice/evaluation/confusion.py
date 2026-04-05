"""Confusion matrix helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_frame(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    """Return a labeled confusion matrix as a DataFrame."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)
