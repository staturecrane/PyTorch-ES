from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import random
import sys

# from evostra import EvolutionStrategy
from evolution import strategies
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

gym_logger.setLevel(logging.CRITICAL)

# add the model on top of the convolutional base
model = nn.Sequential(
    nn.Linear(4, 100),
    nn.ReLU(True),
    nn.Linear(100, 2),
    nn.Softmax(1)
)

pool = ThreadPool(100)

def get_reward(weights, model, render=False):
    # model.set_weights(weights)
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    env = gym.make("CartPole-v0")
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
        batch = ob[np.newaxis,...]
        prediction = cloned_model(Variable(torch.from_numpy(batch).float()))
        action = np.argmax(prediction.data)
        ob, reward, done, _ = env.step(action)

        total_reward += reward 

    env.close()
    return total_reward
    
partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

es = strategies.EvolutionModule(mother_parameters, partial_func, population_size=10, sigma=0.1, learning_rate=0.001)
final_weights = es.run(200)

reward = partial_func(final_weights, render=True)
print(f"Reward from final weights: {reward}")