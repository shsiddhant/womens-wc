from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from typing import Literal
    from numpy.typing import NDArray


def accuracy(
    conf_matrix: NDArray[np.int32]
    ):
    if not conf_matrix.sum():
        raise ValueError("Sum of confusion matrix cannot be less than 1")
    else:
        return conf_matrix.diagonal().sum() / conf_matrix.sum()

def precision_recall(
    conf_matrix: NDArray[np.int32],
    ):
    """
    Calculate precision and recall from confusion matrix.
    """
    shape = conf_matrix.shape
    if (
        len(shape) !=2 or
        shape[0] != shape[1] or
        shape[0] <=0
    ):
        raise ValueError("'conf_matrix' must be a square matrix of non-zero size.")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            metrics =  np.array([
                conf_matrix.diagonal() / conf_matrix.sum(axis=axis) for axis in [0, 1]
                ])
        return metrics

