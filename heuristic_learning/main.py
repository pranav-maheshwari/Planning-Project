#!/usr/bin/env python
import numpy as np
from environment_database import *

#Set the learning parameters
total_episodes = 3000
learning_rate = 0.01
episode_length = 5000


#Initialize the weights buffer
w_buffer = [(1.0 ,0.0, 0.0, 0.0)]

env_database = getEnvironmentDatabase()


