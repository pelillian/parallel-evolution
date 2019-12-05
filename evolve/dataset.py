"""
This module implements the evolutionary algorithm.
"""

import numpy as np

from evolve.model import get_model
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

train_samples = 5000
def get_mnist():
  X, y = fetch_openml('mnist_784', cache=True, version=1, return_X_y=True, data_home='./dataset')
  random_state = check_random_state(0)
  permutation = random_state.permutation(X.shape[0])
  X = X[permutation]
  y = y[permutation]
  y = y.astype(np.int)
  X = X.reshape((X.shape[0], -1))
  X /= 255

  num_classes = np.max(y) + 1
#  y = np.eye(num_classes)[y]
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=10000)
  return X_train, X_test, y_train, y_test, num_classes

