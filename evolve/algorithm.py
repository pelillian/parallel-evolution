"""
This module implements the evolutionary algorithm.
"""

import numpy as np

from evolve.model import get_model
from sklearn.datasets import fetch_openml
from evolve.dataset import get_mnist


def train(model_type, pop_size=10, num_gen=100):
    """Primary train loop."""
    model = get_model(model_type)

    dataset = get_mnist()

    pop_shape = (pop_size,) + model.param_shape(dataset[0].shape)
    population = np.random.rand(*pop_shape)

    for gen in range(num_gen):
        pass

def test(model_type):
    pass

