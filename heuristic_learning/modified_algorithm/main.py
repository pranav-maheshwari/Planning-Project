#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

from environment_database import *
from modified_learner import ModifiedLearner
from graphs.HeuristicFunctions import *

# Set the learning parameters
total_episodes = 100
episode_length = 500
learning_rate = 0.001
batch_size = 32
seed = 1234

#Set the base heuristic value
base_heuristic = Manhattan

#Set training parameters
include_terminal = False
batch_train = True

visualize = True
#Get database of environments to run experiments on
test_env_database = getEnvironmentDatabase()

l = ModifiedLearner(total_episodes, \
            episode_length, \
            learning_rate, \
            batch_size, \
            test_env_database, \
            seed, \
            base_heuristic, \
            include_terminal, \
            batch_train, \
            visualize)
l.learn() 
