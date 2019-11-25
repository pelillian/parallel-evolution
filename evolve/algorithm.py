"""
This module implements the evolutionary algorithm.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from evolve.model import get_model
from evolve.dataset import get_mnist


def train(model_type, pop_size=10, num_gen=100):
    """Primary train loop."""
    X_train, X_test, y_train, y_test = get_mnist()

    model = get_model(model_type, num_classes=max(y_test)+1)

    pop_shape = (pop_size,) + model.param_shape(X_train[0].shape)
    population = np.random.rand(*pop_shape)

    for gen in range(num_gen):
        for individual in population:
            fitness = eval(model, individual, X_train, y_train)

def eval(model, params, X_train, y_train):
    """This method calculates fitness given a model, its parameters, and a dataset."""
    y_pred = model.predict(X_train, params)
    import pdb; pdb.set_trace()
    accuracy = accuracy_score(y_true=y_train, y_pred=y_pred)
    return accuracy

def test(model_type):
    pass

