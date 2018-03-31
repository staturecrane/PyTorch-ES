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
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from torchvision import transforms
gym_logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_path', type=str, required=True, help='Path to save final weights')
parser.add_argument('-c', '--cuda', action='store_true', help='Whether or not to use CUDA')
parser.set_defaults(cuda=False)

args = parser.parse_args()

cuda = args.cuda and torch.cuda.is_available()

model = nn.Sequential(
    nn.Linear(8, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 4),
    nn.Softmax(1)
)

if cuda:
    model = model.cuda()


def get_reward(weights, model, render=False):

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)

    env = gym.make("LunarLander-v2")
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.01)
        batch = torch.from_numpy(ob[np.newaxis,...]).float()
        if cuda:
            batch = batch.cuda()
        prediction = cloned_model(Variable(batch))
        action = prediction.data.numpy().argmax()
        ob, reward, done, _ = env.step(action)

        total_reward += reward 
    env.close()
    return total_reward

weights = pickle.load(open(os.path.abspath(args.weights_path), 'rb'))

print(get_reward(weights, model, render=True))
