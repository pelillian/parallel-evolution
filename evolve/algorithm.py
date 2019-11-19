"""
This module implements the evolutionary algorithm.
"""

import numpy as np

from evolve.model import get_model


def train(model_type, pop_size=10):
    model = get_model(model_type)
    population = np.random.rand(pop_size, 10, 2)

def test(model_type):
    pass

