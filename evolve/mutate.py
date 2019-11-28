"""
This module mutates a given individual.
"""

import numpy as np


def add_noise_to_array(array, mu, sigma):
  # In the normal distribution, mu is the mean, and sigma is the standard deviation
  noise = np.random.normal(mu, sigma, array.shape)
  return array + noise

