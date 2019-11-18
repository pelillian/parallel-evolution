"""
This module contains the models we will use for evolution.
"""

import numpy as np


class Model:
    def __init__(self):
        """Defines a ML predictor for evolution."""
        pass

    def predict(self, X, params):
        """Use model to predict y given X."""
        pass

class LinearModel(Model):
    def __init__(self):
        """Defines a linear predictor."""
        pass

    def predict(self, X, params):
        """Use linear model to predict y given X. Input: np.ndarray."""
        weights = params[:, 0]
        bias = params[:, 1]
        output = X * weights + bias
        y = np.sum(output)
        return y

