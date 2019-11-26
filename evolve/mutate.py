"""
This module mutates the data for each generation
"""

import numpy as np
def add_noise_to_array(mu, array):
  sigma = np.std(array)
  # mu is the mean
  # sigma is the standard deviation of the normal distribution
  noise = np.random.normal(mu, sigma, len(array))
  noisy_array = array + noise
  return noisy_array

