"""
This module implements the evolutionary algorithm.
"""

import numpy as np

from evolve.model import get_model


def train(model_type, pop_size=10, num_gen=100):
    """Primary train loop."""
    model = get_model(model_type)

    X_train, X_test, y_train, y_test = get_mnist()

    pop_shape = (pop_size,) + model.param_shape(X_train[0].shape)
    population = np.random.rand(*pop_shape)

    for gen in range(num_gen):
        for individual in population:
            fitness = eval(model, params, X_train, y_train)

def eval(model, params, X_train, y_train):
    """This method calculates fitness given a model, its parameters, and a dataset."""
    fitness = []
    for X in X_train:
        y_pred = model.predict(X, params)
        #TODO: compare y_pred and y
    mean_fitness = np.mean(fitness)
    return mean_fitness

def test(model_type):
    pass

