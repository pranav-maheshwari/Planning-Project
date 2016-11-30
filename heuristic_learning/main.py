#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from environment_database import *
from learner import Learner
from graphs.HeuristicFunctions import *

# Set the learning parameters
total_episodes = 100
learning_rate = 0.001
episode_length = 500
batch_size = 32
seed = 1234
base_heuristic = Euclidean
lambda_factor = 0.1
num_features = 1
num_epochs = 1
include_terminal = True
env_database = getEnvironmentDatabase()

l = Learner(total_episodes, \
            learning_rate, \
            episode_length, \
            batch_size, \
            num_epochs, \
            env_database, \
            seed, \
            base_heuristic, \
            lambda_factor, \
            num_features, \
            include_terminal, \
            True)
l.learn()
