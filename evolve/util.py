"""
Misc utilities for our algorithm
"""

import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective',
                        default='fit',
                        const='fit',
                        nargs='?',
                        metavar='OBJECTIVE',
                        action='store',
                        choices=['fit', 'predict'],
                        help='Fit vs predict (train vs test)')
    args = parser.parse_args()
    return args

