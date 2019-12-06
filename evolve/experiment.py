"""
This module contains the main method from which our experiments are run.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import dowel
from dowel import logger, tabular
from datetime import datetime
import numpy as np

from evolve.util import read_args
from evolve.algorithm import test
from evolve.dataset import get_mnist


def main():
    args = read_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    if args.workers is not None and args.parallel_strategy is None:
        raise ValueError('To use a defined number of workers, set the parallel-strategy argument')

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

    logger.add_output(dowel.StdOutput())
    logger.add_output(dowel.TextOutput(log))
    logger.log(str(args))

    X_train, X_test, y_train, y_test, num_classes = get_mnist()
    logger.log('Loaded Dataset')

    if args.objective == 'train':
        if args.parallel_strategy is None:
            from evolve.algorithm import train
        if args.parallel_strategy == 'distributed':
            from evolve.distributed import train
        if args.parallel_strategy == 'master-worker':
            from evolve.masterworker import train

        population = train(
                args.model,
                X_train,
                y_train,
                num_classes,
                num_workers=args.workers,
                pop_size=args.population,
                num_gen=args.generations,
                fit_cutoff=args.fitness_cutoff,
                target_accuracy=args.target_accuracy,
                noise_sigma=args.noise_sigma,
                checkpoint=checkpoint_file,
                log_gen=args.log_gen,
                save_gen=args.save_gen,
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
