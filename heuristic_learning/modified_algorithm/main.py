#!/usr/bin/env python
import sys
import os
import pickle
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))

from environment_database import *
from modified_learner import Learner
from graphs.HeuristicFunctions import *

# Set the learning parameters
total_episodes = 100
episode_length = 4000
learning_rate = 0.001
batch_size = 32
seed = 1234

#Set the base heuristic value
base_heuristic = Manhattan

#Set training parameters
include_terminal = False
batch_train = True

visualize = True
graph_connectivity = "four_connected"
num_env_to_load = 101
swamp_cost = 100
load_from_pickle = False
save_to_pickle = True
need_additional_features = False
preloaded = True
#Get database of environments to run experiments on
# if not load_from_pickle:
test_env_database = getEnvironmentDatabase(graph_connectivity, "soft", swamp_cost, num_env_to_load, )
# 	if save_to_pickle:
# 		pickle.dump( test_env_database, open( "save.p", "wb" ) )
# else:
	# test_env_database = pickle.load( open( "save.p", "rb" ))

l = Learner(total_episodes, \
            episode_length, \
            learning_rate, \
            test_env_database, \
            seed, \
            base_heuristic, \
            include_terminal, \
            visualize)

l.learn_batch_mode()

# l2 = Learner(total_episodes, \
#             episode_length, \
#             learning_rate, \
#             test_env_database, \
#             seed, \
#             base_heuristic, \
#             include_terminal, \
#             visualize)

# l2.learn_online_mode()