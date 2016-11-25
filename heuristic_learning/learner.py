#!/usr/bin/env python

import numpy as np

class Learner:
	def __init__(self ,\
				 total_episodes ,\
				 learning_rate ,\
				 episode_length):
	self.total_episodes = total_episodes
	self.learning_rate = learning_rate
	self.episode_length = episode_length
