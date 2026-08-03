import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)+ 1e-7
        t1 = y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)
        l = -1/len(y_true)*np.sum(t1)
        return np.round(l,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)+ 1e-7
        t1 = y_true*np.log(y_pred)
        l = -1/len(y_pred)*np.sum(np.sum(t1))
        return np.round(l,4)