from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import pickle
import random
import sys

# from evostra import EvolutionStrategy
from evolutionary_strategies.strategies.evolution import EvolutionModule
from evolutionary_strategies.utils.helpers import weights_init
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

gym_logger.setLevel(logging.CRITICAL)
cuda = torch.cuda.is_available()

# add the model on top of the convolutional base
model = nn.Sequential(
    nn.Linear(24, 100),
    nn.ReLU(True),
    nn.Linear(100, 100),
    nn.ReLU(True),
    nn.Linear(100, 4),
    nn.Tanh()
)

model.apply(weights_init)

if cuda:
    model = model.cuda()

pool = ThreadPool(100)

def get_reward(weights, model, render=False):

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    env = gym.make('BreakoutDeterministic-v4')    
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
        batch = torch.from_numpy(ob[np.newaxis,...]).float()
        if cuda:
            batch = batch.cuda()
        prediction = cloned_model(Variable(batch, volatile=True))
        action = prediction.data[0]
        ob, reward, done, _ = env.step(action)

        total_reward += reward 

    env.close()
    return total_reward

env = gym.make('BreakoutDeterministic-v4')    
print(env.action_space_sample())
"""
partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

es = strategies.EvolutionModule(mother_parameters, partial_func, population_size=25, sigma=0.1, learning_rate=0.001, threadcount=8)
final_weights = es.run(1000)
pickle.dump(final_weights, open('weights.p', 'wb'))

reward = partial_func(final_weights, render=True)
print(f"Reward from final weights: {reward}")
"""