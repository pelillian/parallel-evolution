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
        pass

def test(model_type):
    pass

