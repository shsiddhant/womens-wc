from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.base import clone, BaseEstimator

if TYPE_CHECKING:
    from typing import Literal
    import pandas as pd
    from sklearn.linear_model import LogisticRegression


def train_validate_test(
    features: pd.DataFrame,
    target: pd.DataFrame,
    cutoffs: pd.DatetimeIndex,
    drop_cols: list[str]
    ):
    Index_train = features.start_date < cutoffs[0]
    Index_test = (
        (features.start_date >= cutoffs[0]) & (features.start_date <= cutoffs[1])
    )
    return (
        features[Index_train].drop(columns=drop_cols),
        target[Index_train].result,
        features[Index_test].drop(columns=drop_cols),
        target[Index_test].result
    )

def log_loss(
    log_prob_matrix: np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]],
    y_test: np.ndarray[tuple[int]] | pd.Series
    ):
    """
    Calculate Log Loss from Log Probability Matrix for binary classification problems.
    """
    loss = - np.mean(
        y_test * log_prob_matrix[:, 1]
        + (1 - y_test) * log_prob_matrix[:, 0]
        )
    return loss

def cross_validation(
    model_input: BaseEstimator,
    X_train: np.ndarray[tuple[int, int]],
    Y_train: pd.Series,
    min_first: int,
    n_parts: int,
    predict_prob: bool = False
    ):
    if not isinstance(n_parts, int) or n_parts <= 0:
        raise ValueError("n_parts must a positive integer.")
    size = len(X_train) // n_parts
    indices = [
        min_first + size * k
        for k in range(1, n_parts, 1)
        if min_first + size * k < len(X_train)
    ]
    conf_matrices = []
    y_pred_values = []
    if predict_prob:
        probs = []
        log_loss_values = []
    else:
        probs = None
        log_loss_values = None
    for i in indices:
        model = clone(model_input)
        x_train, y_train = X_train[:i], Y_train.iloc[:i]
        x_test, y_test = X_train[i:i + size], Y_train.iloc[i:i+size]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred_values.append(y_pred)
        conf_matrices.append(confusion_matrix(y_test, y_pred))
        if probs is not None and log_loss_values is not None:
            model: LogisticRegression
            prob = model.predict_proba(x_test)
            probs.append(prob)
            log_loss_values.append(log_loss(np.log(prob), y_test))
    results = {
            "CV_matrices": conf_matrices,
            "CV_predictions": y_pred_values
    }
    if probs is not None and log_loss_values is not None:
        results["CV_probabilities"] = probs
        results["CV_log_loss"] = log_loss_values
    return results
