"""
This module contains the main method from which our experiments are run.
"""

import numpy as np

from evolve.util import read_args


def train(model_type):
    pass

def test(model_type):
    pass

def main():
    args = read_args()
    if args.objective == 'train':
        train(args.model_type)
    elif args.objective == 'test':
        trest(args.model_type)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
