"""
This module implements the evolutionary algorithm in parallel.
"""

import ray
import numpy as np
from sklearn.metrics import log_loss
from dowel import logger, tabular
from datetime import datetime

from evolve.model import get_model
from evolve.mutate import add_noise_to_array


def train(
            model_type,
            X_train,
            y_train,
            num_classes,
            num_workers=None,
            pop_size=10,
            num_gen=100,
            fit_cutoff=50,
            noise_sigma=0.1,
            checkpoint='checkpoint.npy',
            log_gen=10,
            save_gen=100,
            population=None,
            target_accuracy=None,
        ):
    """Primary train loop."""
    logger.log('Starting Evolutionary Algorithm!')
    ray.init(num_cpus=num_workers)
    logger.log('Initialized Ray')

    algorithm_time = datetime.now()

    model = get_model(model_type, num_classes=num_classes)

    individual_shape = model.param_shape(X_train[0].shape)
    pop_shape = (pop_size,) + individual_shape
    if population is None:
        population = np.random.rand(*pop_shape)
    assert population.shape == pop_shape

    num_fit = int(round( (1 - (fit_cutoff / 100)) * pop_size ))
    if num_fit == pop_size:
        raise ValueError('fit_cutoff too low')

    for gen in range(num_gen):
        start_gen = datetime.now()
        tabular.record('Generation', gen)

        fitness_scores, accuracy_scores = evaluate_population(model, population, X_train, y_train)

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

        gen_time = datetime.now() - start_gen
        tabular.record('Execution Time', gen_time.total_seconds())
        if gen % log_gen == 0 or gen == num_gen - 1:
            logger.log(tabular)
            if target_accuracy is not None and 100 * np.max(accuracy_scores) > target_accuracy:
                logger.log('Stopping early because target accuracy reached')
                return population

        logger.dump_all()

        if gen % save_gen == 0 or gen == num_gen - 1:
            np.save(checkpoint, population)

    return population

def evaluate_population(model, population, X, y):
    scores = ray.get([evaluate_individual.remote(model, individual, X, y) for individual in population])
    scores = np.array(scores)

    fitness_scores = scores[:, 0]
    accuracy_scores = scores[:, 1]
    return fitness_scores, accuracy_scores

@ray.remote
def evaluate_individual(model, params, X, y):
    """This method calculates fitness given a model, its parameters, and a dataset."""
    y_pred = model.predict(X, params)
    fitness = log_loss(y_true=y, y_pred=y_pred)

    y_pred_class = np.argmax(y_pred, axis=-1)
    accuracy = np.sum(y_pred_class == y) / len(y)

    return fitness, accuracy

