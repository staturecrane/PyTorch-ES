import argparse
from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import random
import sys
import time

# from evostra import EvolutionStrategy
from pytorch_es import EvolutionModule
from pytorch_es.utils.helpers import weights_init
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

gym_logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_path', type=str, required=True, help='Path to save final weights')
parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
parser.set_defaults(cuda=False)

args = parser.parse_args()

cuda = args.cuda and torch.cuda.is_available()

# add the model on top of the convolutional base
model = nn.Sequential(
    nn.Linear(24, 100),
    nn.Linear(100, 100),
    nn.Linear(100, 4),
    nn.Tanh()
)

model.apply(weights_init)

if cuda:
    model = model.cuda()

def get_reward(weights, model, render=False):

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)

    env = gym.make("BipedalWalkerHardcore-v2")
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
    
partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

es = EvolutionModule(
    mother_parameters, partial_func, population_size=50, 
    sigma=0.1, learning_rate=0.001, reward_goal=300, consecutive_goal_stopping=20,
    threadcount=8, cuda=cuda, render_test=True
)
start = time.time()
final_weights = es.run(4000, print_step=1)
end = time.time() - start

pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))

reward = partial_func(final_weights, render=True)

print(f"Reward from final weights: {reward}")
print(f"Time to completion: {end}")