"""
This module implements the evolutionary algorithm.
"""

import numpy as np
from sklearn.metrics import log_loss
from dowel import logger, tabular

from evolve.model import get_model
from evolve.dataset import get_mnist
from evolve.mutate import add_noise_to_array


def train(model_type, pop_size=10, num_gen=100, fit_cutoff=70, noise_sigma=0.1, checkpoint='checkpoint.npy', population=None):
    """Primary train loop."""
    X_train, X_test, y_train, y_test, num_classes = get_mnist()
    logger.log('Loaded Dataset')

    model = get_model(model_type, num_classes=num_classes)

    individual_shape = model.param_shape(X_train[0].shape)
    pop_shape = (pop_size,) + individual_shape
    if population is None:
        population = np.random.rand(*pop_shape)
    assert population.shape == pop_shape

    fitness_scores = np.zeros(pop_size)
    accuracy_scores = np.zeros(pop_size)
    num_fit = int(round( (1 - (fit_cutoff / 100)) * pop_size ))
    if num_fit == pop_size:
        raise ValueError('fit_cutoff too low')

    for gen in range(num_gen):
        tabular.record('Generation', gen)

        for idx, individual in enumerate(population):
            fitness, accuracy = evaluate(model, individual, X_train, y_train)
            fitness_scores[idx] = fitness
            accuracy_scores[idx] = accuracy

        fit_idx = fitness_scores.argsort()[:num_fit]
        unfit_idx = fitness_scores.argsort()[num_fit:]
        fit_individuals = population[fit_idx]
        assert 0 < len(fit_individuals) < pop_size

        tabular.record('Fitness Best', np.min(fitness_scores))
        tabular.record('Fitness Mean', np.mean(fitness_scores))
        tabular.record('Accuracy Best', np.max(accuracy_scores))
        tabular.record('Accuracy Mean', np.mean(accuracy_scores))

        for idx in unfit_idx:
            choice = np.random.choice([0, 1, 2], p=[0.8, 0.2, 0])
            sigma = np.abs(np.random.normal(0, noise_sigma))
            if choice == 0:
                random_fit_individual = fit_individuals[np.random.choice(len(fit_individuals))]
                population[idx] = add_noise_to_array(random_fit_individual, mu=0, sigma=sigma)
            elif choice == 1:
                random_individual = population[np.random.choice(len(population))]
                population[idx] = add_noise_to_array(random_individual, mu=0, sigma=sigma)
            else:
                population[idx] = np.random.rand(*individual_shape)

        if gen % 10 == 0:
            logger.log(tabular)
        logger.dump_all()

        if gen % 100 == 0:
            np.save(checkpoint, population)

def evaluate(model, params, X_train, y_train):
    """This method calculates fitness given a model, its parameters, and a dataset."""
    y_pred = model.predict(X_train, params)
    fitness = log_loss(y_true=y_train, y_pred=y_pred)

    y_pred_class = np.argmax(y_pred, axis=-1)
    accuracy = np.sum(y_pred_class == y_train) / len(y_train)

    return fitness, accuracy

def test(model_type):
    pass

