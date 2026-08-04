import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        X = np.array(X)
        print(X.shape)
        print(weights.shape)

        y = np.dot(X,weights)
    
        return np.round(y,5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        model_prediction = np.array(model_prediction)
        ground_truth = np.array(ground_truth)
        t1 = np.sum((model_prediction - ground_truth)**2)
        mse = 1/len(model_prediction) * t1
        return np.round(mse,5)

