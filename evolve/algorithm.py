"""
This module implements the evolutionary algorithm.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from dowel import logger, tabular

from evolve.model import get_model
from evolve.dataset import get_mnist
from evolve.mutate import add_noise_to_array


def train(model_type, pop_size=10, num_gen=100, fit_cutoff=60, noise_sigma=0.5):
    """Primary train loop."""
    X_train, X_test, y_train, y_test = get_mnist()
    logger.log('Loaded Dataset')

    model = get_model(model_type, num_classes=max(y_test)+1)

    pop_shape = (pop_size,) + model.param_shape(X_train[0].shape)
    population = np.random.rand(*pop_shape)
    fitness_scores = np.zeros(pop_size)

    for gen in range(num_gen):
        logger.push_prefix('gen {}'.format(gen))

        for idx, individual in enumerate(population):
            fitness = eval(model, individual, X_train, y_train)
            fitness_scores[idx] = fitness

        cutoff = np.percentile(fitness_scores, fit_cutoff)
        fit_individuals = population[np.argwhere(fitness_scores > cutoff)]

        tabular.record('Best Individual', np.max(fitness_scores))
        tabular.record('Mean Individual', np.mean(fitness_scores))

        for idx in range(len(population)):
            if fitness_scores[idx] < cutoff:
                random_fit_individual = fit_individuals[np.random.choice(len(fit_individuals))]
                population[idx] = add_noise_to_array(random_fit_individual, sigma=noise_sigma)

        logger.log(tabular)
        logger.pop_prefix()
        logger.dump_all()

def eval(model, params, X_train, y_train):
    """This method calculates fitness given a model, its parameters, and a dataset."""
    y_pred = model.predict(X_train, params)
    accuracy = accuracy_score(y_true=y_train, y_pred=y_pred)
    return accuracy

def test(model_type):
    pass

