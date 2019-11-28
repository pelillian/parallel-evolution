"""
This module contains the main method from which our experiments are run.
"""

import dowel
from dowel import logger, tabular

from evolve.util import read_args
from evolve.algorithm import train, test


def main():
    logger.add_output(dowel.StdOutput())
    logger.log('Starting Evolutionary Algorithm!')

    args = read_args()
    if args.objective == 'train':
        train(args.model, pop_size=args.population, num_gen=args.generations, fit_cutoff=args.fitness_cutoff, noise_sigma=args.noise_sigma)
    elif args.objective == 'test':
        test(args.model)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
