"""
Misc utilities for our algorithm
"""

import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='train', choices=['train', 'test'], help='Objective: train or test?')
    parser.add_argument('--model', default='linear', choices=['linear', 'neural'], help='Type of model to use for the experiment')
    parser.add_argument('--population', default=10, type=int, metavar='[0-2000]', choices=range(2000), help='Size of the training population')
    parser.add_argument('--generations', default=100, type=int, metavar='[0-50000]', choices=range(50000), help='Number of generations to run experiment')
    parser.add_argument('--fitness_cutoff', default=60, help='Population fitness percentage to replace each generation')
    parser.add_argument('--noise_sigma', default=2, help='Standard deviation of the mutation noise vector')
    args = parser.parse_args()
    return args

