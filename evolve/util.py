"""
Misc utilities for our algorithm
"""

import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='train', choices=['train', 'test'], help='Objective: train or test?')
    parser.add_argument('--model', default='linear', choices=['linear', 'neural'], help='Type of model to use for the experiment')
    parser.add_argument('--population', default=10, type=int, help='Size of the training population')
    parser.add_argument('--generations', default=100, type=int, help='Number of generations to run experiment')
    parser.add_argument('--fitness_cutoff', default=70, type=int, help='Population fitness percentage to replace each generation')
    parser.add_argument('--noise_sigma', default=2, type=int, help='Standard deviation of the mutation noise vector')
    parser.add_argument('--name', type=str, help='Experiment name')
    args = parser.parse_args()
    return args

