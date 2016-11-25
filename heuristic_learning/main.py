#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from environment_database import *
from learner import Learner
from graphs.HeuristicFunctions import *
#Set the learning parameters
total_episodes = 3000
learning_rate = 0.01
episode_length = 5000
batch_size = 4
seed = 1234
base_heuristic = Manhattan 
env_database = getEnvironmentDatabase()

l = Learner(total_episodes,\
			 learning_rate,\
			 episode_length,\
			 batch_size,\
			 env_database,\
			 seed,\
			 Manhattan)
l.learn()
print("Learner Initialized")
