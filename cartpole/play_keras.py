from collections import deque
import gc
import random

from evostra import EvolutionStrategy
import gym
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
from PIL import Image
import skimage.transform
import tensorflow as tf


env = gym.make("CartPole-v0")
# add the model on top of the convolutional base
inputs = Input(shape=(4,))
x = Dense(100, activation='relu')(inputs)
prediction = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=prediction)
model.load_weights('cartpole/cartpole.h5')

def play():
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        batch = ob[np.newaxis,...]
        prediction = model.predict(batch)
        action = np.argmax(prediction)
        ob, reward, done, _ = env.step(action)

        total_reward += reward 

    return total_reward

reward = play()
print(f'Reward: {reward}')