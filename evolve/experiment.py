"""
This module contains the main method from which our experiments are run.
"""

import os
import dowel
from dowel import logger, tabular
from datetime import datetime
import numpy as np

from evolve.util import read_args
from evolve.algorithm import train, test
from evolve.dataset import get_mnist


def main():
    args = read_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    filename = args.model + '_' + datetime.now().strftime("%Y-%b-%d-%H:%M:%S")
    if args.name:
        filename = args.name + '_' + filename

    population = None

    if args.objective == 'test':
        filename = 't_' + args.checkpoint
    elif args.checkpoint:
        filename = 'r_' + args.checkpoint

    checkpoint_file = os.path.join('checkpoints', filename + '.npy')
    log = os.path.join('logs', filename + '.log')

    if args.checkpoint:
        population = np.load(os.path.join('checkpoints', args.checkpoint + '.npy'))

    X_train, X_test, y_train, y_test, num_classes = get_mnist()
    logger.log('Loaded Dataset')

    logger.add_output(dowel.StdOutput())
    logger.add_output(dowel.TextOutput(log))
    logger.log('Starting Evolutionary Algorithm!')
    logger.log(str(args))

    if args.objective == 'train':
        population = train(
                args.model,
                X_train,
                y_train,
                num_classes,
                pop_size=args.population,
                num_gen=args.generations,
                fit_cutoff=args.fitness_cutoff,
                noise_sigma=args.noise_sigma,
                checkpoint=checkpoint_file,
                population=population,
             )
        test(args.model, population, X_test, y_test, num_classes)
    elif args.objective == 'test':
        if not args.checkpoint:
            raise ValueError('checkpoint arg must be set in order to test algorithm')
        test(args.model, population, X_test, y_test, num_classes)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
