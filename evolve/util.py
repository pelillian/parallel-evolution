"""
Misc utilities for our algorithm
"""

import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='train', choices=['train', 'test'], help='Objective: train or test?')
    parser.add_argument('--model', default='linear', choices=['linear'], help='Type of model to use for the experiment')
    parser.add_argument('--population', default=512, type=int, help='Size of the training population')
    parser.add_argument('--generations', default=5000, type=int, help='Number of generations to run experiment')
    parser.add_argument('--fitness_cutoff', default=50, type=int, help='Population fitness percentage to replace each generation')
    parser.add_argument('--noise_sigma', default=0.1, type=int, help='Standard deviation of the mutation noise vector')
    parser.add_argument('--target_accuracy', type=int, help='Accuracy percentage to end training')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--checkpoint', type=str, help='Population checkpoint to resume training')
    parser.add_argument('--log_gen', default=10, type=int, help='Number of generations before we log values')
    parser.add_argument('--save_gen', default=100, type=int, help='Number of generations before we checkpoint')
    parser.add_argument('--workers', default=None, type=int, help='Number of parallel workers to spawn')
    parser.add_argument('--parallel_strategy', choices=[None, 'distributed', 'master-worker'], help='The strategy to use during parallelization')
    args = parser.parse_args()
    return args

