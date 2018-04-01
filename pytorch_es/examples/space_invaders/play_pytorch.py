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

num_features = 4
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

class BreakoutModel(nn.Module):
    def __init__(self, num_features):
        super(BreakoutModel, self).__init__()        
        self.main = nn.Sequential(
            # input size is in_size x 64 x 64
            nn.Conv2d(3, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 32 x 32
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 2) x 16 x 16
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 4) x 8 x 8
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 8, num_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf * 8) x 4 x 4
            nn.Conv2d(num_features * 16, 6, 4, 1, 0, bias=False),
            nn.Softmax(1)
        )


    def initialize_hidden(self, cuda):
        if cuda:
             return (Variable(torch.randn(1, 1, 50).cuda()),
                    Variable(torch.randn(1, 1, 50).cuda()))
        return (Variable(torch.randn(1, 1, 50)),
                Variable(torch.randn((1, 1, 50))))

    
    def forward(self, input): #, prev_hidden):
        main = self.main(input)
        return main


model = BreakoutModel(num_features)

if cuda:
    model = model.cuda()


def get_reward(weights, model, render=False, save=False):

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    env = gym.make("SpaceInvaders-v0")
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.005)
        image = transform(Image.fromarray(ob))
        image = image.unsqueeze(0)
        if cuda:
            image = image.cuda()
        prediction = cloned_model(Variable(image, volatile=True))
        action = np.argmax(prediction.data)
        ob, reward, done, _ = env.step(action)

        total_reward += reward 
    env.close()
    return total_reward

weights = pickle.load(open(os.path.abspath(args.weights_path), 'rb'))

get_reward(weights, model, render=True)