"""
This module implements the evolutionary algorithm.
"""

import numpy as np
from sklearn.metrics import accuracy_score
from dowel import logger, tabular

from evolve.model import get_model
from evolve.dataset import get_mnist
from evolve.mutate import add_noise_to_array


def train(model_type, pop_size=10, num_gen=100, fit_cutoff=60, noise_sigma=2, checkpoint='checkpoint.npy'):
    """Primary train loop."""
    X_train, X_test, y_train, y_test = get_mnist()
    logger.log('Loaded Dataset')

    model = get_model(model_type, num_classes=max(y_test)+1)

    individual_shape = model.param_shape(X_train[0].shape)
    pop_shape = (pop_size,) + individual_shape
    population = np.random.rand(*pop_shape)
    fitness_scores = np.zeros(pop_size)
    num_fit = int((1 - (fit_cutoff / 100)) * pop_size)

    for gen in range(num_gen):
        tabular.record('Generation', gen)

        for idx, individual in enumerate(population):
            fitness = eval(model, individual, X_train, y_train)
            fitness_scores[idx] = fitness

        fit_sorted = fitness_scores.argsort()
        fit_idx = fit_sorted[-num_fit:][::-1]
        unfit_idx = fit_sorted[:-num_fit][::-1]
        fit_individuals = population[fit_idx]
        assert len(fit_individuals) > 0

        tabular.record('Fitness Best', np.max(fitness_scores))
        tabular.record('Fitness Mean', np.mean(fitness_scores))

        for idx in unfit_idx:
            choice = np.random.choice([0, 1, 2], p=[0.55, 0.35, 0.1])
            if choice == 0:
                random_fit_individual = fit_individuals[np.random.choice(len(fit_individuals))]
                population[idx] = add_noise_to_array(random_fit_individual, mu=0, sigma=noise_sigma)
            elif choice == 1:
                random_individual = population[np.random.choice(len(population))]
                population[idx] = add_noise_to_array(random_individual, mu=0, sigma=noise_sigma)
            else:
                population[idx] = np.random.rand(*individual_shape)

        logger.log(tabular)
        logger.dump_all()
        
        if gen % 100 == 0:
            np.save(checkpoint, population)

def eval(model, params, X_train, y_train):
    """This method calculates fitness given a model, its parameters, and a dataset."""
    y_pred = model.predict(X_train, params)
    accuracy = accuracy_score(y_true=y_train, y_pred=y_pred)
    return accuracy

def test(model_type):
    pass

