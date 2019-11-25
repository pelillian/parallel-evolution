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

    def param_shape(self, X_shape):
        pass

class LinearModel(Model):
    def __init__(self, num_classes):
        """Defines a linear predictor."""
        self.output_size = num_classes

    def predict(self, X, params):
        """Use linear model to predict y given X. Input: np.ndarray."""
        y = np.zeros((self.output_size, ) + (X.shape[0], ))
        for class_i in range(self.output_size):
            weights = params[class_i, 0]
            bias = params[class_i, 1]
            output = X * weights + bias
            y_cls = np.sum(output, axis=-1) / np.prod(X[0].shape)
            y[class_i] = y_cls
        return np.argmax(y, axis=0)

    def param_shape(self, X_shape):
        if type(X_shape) is not tuple:
            X_shape = (X_shape,)
        return (self.output_size, 2,) + X_shape

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

