"""
This module contains the main method from which our experiments are run.
"""

import os
import dowel
from dowel import logger, tabular
from datetime import datetime

from evolve.util import read_args
from evolve.algorithm import train, test


def main():
    args = read_args()

    filename = args.model + '_' + datetime.now().strftime("%Y-%b-%d-%H:%M:%S")
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = os.path.join('checkpoints', filename + '.npy')
    os.makedirs('logs', exist_ok=True)
    log = os.path.join('logs', 'logs' + filename + '.log')

    logger.add_output(dowel.StdOutput())
    logger.add_output(dowel.TextOutput(log))
    logger.log('Starting Evolutionary Algorithm!')

    if args.objective == 'train':
        train(args.model, pop_size=args.population, num_gen=args.generations, fit_cutoff=args.fitness_cutoff, noise_sigma=args.noise_sigma, checkpoint=checkpoint)
    elif args.objective == 'test':
        test(args.model)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
