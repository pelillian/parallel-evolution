"""
This module implements the evolutionary algorithm.
"""

import numpy as np

from evolve.model import get_model


def train(model_type, pop_size=10, num_gen=100):
    """Primary train loop."""
    model = get_model(model_type)

    #TODO: Replace this placeholder with MNIST dataset
    dataset = np.random.rand(2000, 700)

    pop_shape = (pop_size,) + model.param_shape(dataset[0].shape)
    population = np.random.rand(*pop_shape)

    for gen in range(num_gen):
        for individual in population:
            fitness = eval(model, params, dataset)

def eval(model, params, dataset):
    """This method calculates fitness given a model, its parameters, and a dataset."""
    fitness = []
    for X in dataset:
        y_pred = model.predict(X, params)
        #TODO: compare y_pred and y
    mean_fitness = np.mean(fitness)
    return mean_fitness

def test(model_type):
    pass

