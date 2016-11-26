#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from utils.planner_utils import *
import numpy as np
from search_agent import SearchAgent

class Learner:
	def __init__(self ,\
				 total_episodes ,\
				 learning_rate ,\
				 episode_length ,\
				 batch_size ,\
				 env_database ,\
				 seed ,\
				 base_heuristic):
		self.total_episodes = total_episodes
		self.learning_rate = learning_rate
		self.episode_length = episode_length
		self.batch_size = batch_size
		self.weights_buffer = [(0,0,0,1)]
		self.env_database = env_database
		self.base_heuristic = base_heuristic
		np.random.seed(seed)
	
	def learn(self, visualize=True):
		planning_prob = self.sampleFromDatabase()
		w_mix = self.sampleMixture()
		w = self.learningBestFirstSearch(w_mix, planning_prob, visualize)
		self.weights_buffer.append(w)
		print self.weights_buffer

	def sampleFromDatabase(self):
		idx = np.random.randint(0, len(self.env_database))
		return self.env_database[idx]
	
	def sampleMixture(self):
		#[TODO: Figure out best way to sample from an ML perspective]
		idx = np.random.randint(0, len(self.weights_buffer))
		return np.asarray(self.weights_buffer[idx])

	def learningBestFirstSearch(self, w_mix, planning_prob, visualize=True):
		graph = planning_prob[0]
		start_list = planning_prob[1]
		goals_list = planning_prob[2]
		t = 0
		s = SearchAgent(graph, start_list, goals_list, self.base_heuristic, visualize)
		curr_weights = w_mix
		while t < self.episode_length:
			done, parent, child, feature_vec = s.step(curr_weights, self.base_heuristic)
			if done:
				print("Episode Finished")
				break
			# print(t)
			t += 1
		return tuple(w_mix)




