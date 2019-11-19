"""
This module contains the main method from which our experiments are run.
"""

import numpy as np

from evolve.util import read_args
from evolve.model import get_model


def train(model_type, population=10):
    model = get_model(model_type)
    population = np.random.rand(

def test(model_type):
    pass

def main():
    args = read_args()
    if args.objective == 'train':
        train(args.model, args.population)
    elif args.objective == 'test':
        test(args.model)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
