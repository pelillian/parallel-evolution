"""
This module contains the models we will use for evolution.
"""

import numpy as np
from scipy.special import softmax


class Model:
    def __init__(self):
        """Defines a ML predictor for evolution."""
        pass

    def predict(self, X, params):
        """Use model to predict y given X."""
        pass

    def param_shape(self, X_shape):
        pass

class LinearModel(Model):
    def __init__(self, num_classes):
        """Defines a linear predictor."""
        self.output_size = num_classes

    def predict(self, X, params):
        """Use linear model to predict y given X. Input: np.ndarray."""
        y = np.zeros((self.output_size, ) + (X.shape[0], ))
        weights = params[1:, :]
        bias = params[0, :]
        output = X @ weights + bias
        output = softmax(output, axis=-1)
        return output

    def param_shape(self, X_shape):
        if type(X_shape) is not tuple:
            X_shape = (X_shape + 1,)
        else:
            X_shape = (X_shape[0] + 1,)
        return X_shape + (self.output_size, )

class NeuralModel(Model):
    def __init__(self):
        """Defines a neural network predictor."""
        raise NotImplementedError

    def predict(self, X, params):
        """Use neural model to predict y given X. Input: np.ndarray."""
        pass

    def param_shape(self, X_shape):
        pass

def get_model(model_type, **kwargs):
    if model_type == 'linear':
        return LinearModel(**kwargs)
    elif model_type == 'neural':
        return NeuralModel(**kwargs)

