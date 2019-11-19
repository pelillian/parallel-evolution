"""
Misc utilities for our algorithm
"""

import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default='train', choices=['train', 'test'], help='Objective: train or test?')
    parser.add_argument('--model', default='linear', choices=['linear', 'neural'], help='Type of model to use for the experiment')
    parser.add_argument('--population', default=10, choices=range(2000), help='Size of the training population')
    parser.add_argument('--generations', default=100, choices=range(50000), help='Number of generations to run experiment')
    args = parser.parse_args()
    return args

