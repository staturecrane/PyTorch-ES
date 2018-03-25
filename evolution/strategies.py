"""
Evolutionary Strategies module for PyTorch models -- modified from https://github.com/alirezamika/evostra
"""
import copy
from multiprocessing.pool import ThreadPool

import numpy as np
import torch


class EvolutionModule:

    def __init__(self, weights, reward_func, population_size=50, sigma=0.1, learning_rate=0.001, decay=1.0, threadcount=4):
        np.random.seed(0)
        self.weights = weights
        self.reward_function = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.decay = decay
        self.pool = ThreadPool(threadcount)


    def jitter_weights(self, weights, population=[], no_jitter=False):
        new_weights = []
        for i, param in enumerate(weights):
            if not no_jitter:
                jittered = self.SIGMA * population[i]
                new_weights.append(param.data + torch.from_numpy(jittered).float())
            else:
                new_weights.append(param.data)
        return new_weights



    def run(self, iterations, print_step=10):
        for iteration in range(iterations):

            population = []
            for _ in range(self.POPULATION_SIZE):
                x = []
                for param in self.weights:
                    x.append(np.random.randn(*param.data.size()))
                population.append(x)

            rewards = self.pool.map(
                self.reward_function, 
                [self.jitter_weights(copy.deepcopy(self.weights), population=pop) for pop in population]
            )

            normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)

            for index, param in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                param.data = param.data + (self.LEARNING_RATE / (len(population) * self.SIGMA) * torch.from_numpy(np.dot(A.T, normalized_rewards).T).float())

            self.LEARNING_RATE *= self.decay

            if (iteration+1) % print_step == 0:
                print('iter %d. reward: %f' % (iteration+1, self.reward_function(
                    self.jitter_weights(copy.deepcopy(self.weights), no_jitter=True), render=True)
                    )
                )

        return self.weights