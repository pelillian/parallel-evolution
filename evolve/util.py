"""
Misc utilities for our algorithm
"""

import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective',
                        default='train',
                        const='train',
                        nargs='?',
                        metavar='OBJECTIVE',
                        action='store',
                        choices=['train', 'test'],
                        help='Objective: train or test?')
    parser.add_argument('--model_type',
                        default='linear',
                        const='linear',
                        nargs='?',
                        metavar='MODEL_TYPE',
                        action='store',
                        choices=['linear', 'nn'],
                        help='Type of model to use for the experiment')
    args = parser.parse_args()
    return args

