#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import random
import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import DataPreprocessing
from search_agent import SearchAgent
from utils.planner_utils import *
print("Import done")
class Learner:
	def __init__(self ,\
				 total_episodes ,\
				 learning_rate ,\
				 episode_length ,\
				 batch_size ,\
				 num_epochs ,\
				 env_database ,\
				 seed ,\
				 base_heuristic ,\
				 lambda_factor ,\
				 num_features ,\
				 include_terminal):
		self.total_episodes = total_episodes
		self.learning_rate = learning_rate
		self.episode_length = episode_length
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.weights_buffer = [(0,0,0,0)]
		self.env_database = env_database
		self.base_heuristic = base_heuristic
		self.lambda_factor = lambda_factor
		self.num_features = num_features
		self.include_terminal = include_terminal
		np.random.seed(seed)
		#Save episode data for training
		self.parent_database = []
		self.child_database = []
		self.feature_database = []
		self.best_cost_database = []
		self.e_dot_database = [] 
		
		#Tensorflow stuff
		tf.set_random_seed(seed)
		rng = np.random
		self.sess = tf.Session()
		#Data preprocessing (Standardizing the data)
		data_prep = DataPreprocessing()
		data_prep.add_featurewise_zero_center()
		data_prep.add_featurewise_stdnorm()
		#tf Graph Input
		# self.X = tf.placeholder("float", [None, self.num_features])
		# self.Y = tf.placeholder("float", [None])
		self.input_ = tflearn.input_data(shape = [None, self.num_features],\
									data_preprocessing = data_prep)
		print self.input_
		# self.linear = tflearn.single_unit(self.input_)
		# print self.linear
		self.linear = tflearn.fully_connected(self.input_, 1)
		self.regression = tflearn.regression(self.linear,\
							optimizer = 'sgd',\
							loss = 'mean_square',\
							learning_rate = self.learning_rate,\
							batch_size = self.batch_size)
		print(self.regression)
		self.m = tflearn.DNN(self.regression)
		print(self.m)
		#Set the model weights
		# self.W = tf.Variable(tf.random_normal([self.num_features, 1]))
		# self.b = tf.Variable(tf.random_normal([1]))
		# print(self.W)
		# print(self.b)
		# # #Add ZCA Whitening to input data
		# # #Construct the linear model
		# self.pred = tf.add(tf.mul(self.input_,self.W), self.b)
		# print self.pred
		# #Mean squared error

	def learn(self, visualize=True):
		curr_episode = 0
		while curr_episode < self.total_episodes: 
			planning_prob = self.sampleFromDatabase()
			w_mix = self.sampleMixture()
			w = self.learningBestFirstSearch(w_mix, planning_prob, visualize)
			self.weights_buffer.append(w)
			curr_episode += 1
		print self.weights_buffer

	def sampleFromDatabase(self):
		idx = np.random.randint(0, len(self.env_database))
		return self.env_database[idx]
	
	def sampleMixture(self):
		#[TODO: Figure out best way to sample from an ML perspective]
		coin_toss = random.random()
		if coin_toss < self.lambda_factor or len(self.weights_buffer) == 1:
			return np.asarray(self.weights_buffer[0])
		else:
			idx = np.random.randint(1, len(self.weights_buffer))
			return np.asarray(self.weights_buffer[idx])

	def learningBestFirstSearch(self, w_mix, planning_prob, visualize=True):
		graph = planning_prob[0]
		start_list = planning_prob[1]
		goals_list = planning_prob[2]
		t = 0
		s = SearchAgent(graph, start_list, goals_list, self.base_heuristic, visualize)
		curr_weights = w_mix
		#Reset episode database before start of episode
		self.parent_database = []
		self.child_database = []
		self.feature_database = []
		self.best_cost_database = []
		self.e_dot_database = []
		print("Start New Episode")
		while t < self.episode_length:
			done, parent, child, feature_vec, best_cost, e_dot = s.step(curr_weights, self.base_heuristic) 
			if child is None or feature_vec is None:
				# print(parent, child, feature_vec, e_dot)
				continue;
			t += 1
			if done:
				if self.include_terminal:
					self.parent_database.append(parent)
					self.child_database.append(child)
					self.feature_database.append(feature_vec.tolist())
					self.best_cost_database.append(best_cost)
					self.e_dot_database.append([e_dot])
				print("Episode Finished")
				break
			else:
				self.parent_database.append(parent)
				self.child_database.append(child)
				self.feature_database.append(feature_vec.tolist())
				self.best_cost_database.append(best_cost)
				self.e_dot_database.append([e_dot]) #tflearn wants this		
		print("Initiate learning")
		print(np.asarray(self.feature_database).shape)
		print(np.asarray(self.e_dot_database).shape)
		self.m.fit(np.asarray(self.feature_database), np.asarray(self.e_dot_database), n_epoch = self.num_epochs, show_metric=True, snapshot_epoch=False)
		# print("Normalized inputs were", self.input_.get)
		print("\nRegression result:")
		print("Y = " + str(self.m.get_weights(self.linear.W)) +
				"*X + " + str(self.m.get_weights(self.linear.b)))
		w_learnt = self.m.get_weights(self.linear.W)
		return tuple(w_learnt)





