"""
This module contains the main method from which our experiments are run.
"""

from evolve.util import read_args
from evolve.algorithm import train, test


def main():
    args = read_args()
    if args.objective == 'train':
        train(args.model, pop_size=args.population)
    elif args.objective == 'test':
        test(args.model)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
