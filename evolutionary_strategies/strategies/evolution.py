"""
Evolutionary Strategies module for PyTorch models -- modified from https://github.com/alirezamika/evostra
"""
import copy
from multiprocessing.pool import ThreadPool
import pickle
import time

import numpy as np
import torch


class EvolutionModule:

    def __init__(
        self, 
        weights, 
        reward_func,
        population_size=50,
        sigma=0.1,
        learning_rate=0.001,
        decay=1.0,
        sigma_decay=1.0,
        threadcount=4,
        render_test=False,
        cuda=False,
        reward_goal=None,
        consecutive_goal_stopping=None,
        save_path=None
    ):
        np.random.seed(int(time.time()))
        self.weights = weights
        self.reward_function = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.cuda = cuda
        self.decay = decay
        self.sigma_decay = sigma_decay
        self.pool = ThreadPool(threadcount)
        self.render_test = render_test
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0
        self.save_path = save_path


    def jitter_weights(self, weights, population=[], no_jitter=False):
        new_weights = []
        for i, param in enumerate(weights):
            if no_jitter:
                new_weights.append(param.data)
            else:
                jittered = torch.from_numpy(self.SIGMA * population[i]).float()
                if self.cuda:
                    jittered = jittered.cuda()
                new_weights.append(param.data + jittered)
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
            if np.std(rewards) != 0:
                normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                for index, param in enumerate(self.weights):
                    A = np.array([p[index] for p in population])
                    rewards_pop = torch.from_numpy(np.dot(A.T, normalized_rewards).T).float()
                    if self.cuda:
                        rewards_pop = rewards_pop.cuda()
                    param.data = param.data + self.LEARNING_RATE / (self.POPULATION_SIZE * self.SIGMA) * rewards_pop

                    self.LEARNING_RATE *= self.decay
                    self.SIGMA *= self.sigma_decay

            if (iteration+1) % print_step == 0:
                test_reward = self.reward_function(
                    self.jitter_weights(copy.deepcopy(self.weights), no_jitter=True), render=self.render_test
                )
                print('iter %d. reward: %f' % (iteration+1, test_reward))

                if self.save_path:
                    pickle.dump(self.weights, open(self.save_path, 'wb'))
                
                if self.reward_goal and self.consecutive_goal_stopping:
                    if test_reward >= self.reward_goal:
                        self.consecutive_goal_count += 1
                    else:
                        self.consecutive_goal_count = 0

                    if self.consecutive_goal_count >= self.consecutive_goal_stopping:
                        return self.weights

        return self.weights