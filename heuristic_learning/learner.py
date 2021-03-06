#!/usr/bin/env python
import os
import sys
import cv2
from threading import Thread

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import random


import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import DataPreprocessing

from search_agent import SearchAgent
from utils.planner_utils import *
from sklearn import linear_model

print("Import done")


class Learner:
    def __init__(self, \
                 total_episodes, \
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
                 visualize):
        self.total_episodes = total_episodes
        self.learning_rate = learning_rate
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weights_buffer = [(0, 0)]
        self.env_database = env_database
        self.base_heuristic = base_heuristic
        self.lambda_factor = lambda_factor
        self.num_features = num_features
        self.include_terminal = include_terminal
        # Visualization related
        self.visualize = visualize
        self.img = np.array([0])
        self.t1 = Thread(target=self.display)

        np.random.seed(seed)
        # Save episode data for training
        self.parent_database = []
        self.child_database = []
        self.feature_database = []
        self.best_cost_database = []
        self.e_dot_database = []
        """
        # Tensorflow stuff
        tf.set_random_seed(seed)
        # rng = np.random
        # self.sess = tf.Session()
        # Data preprocessing (Standardizing the data)
        data_prep = DataPreprocessing()
        data_prep.add_featurewise_zero_center()
        data_prep.add_featurewise_stdnorm()
        # tf Graph Input
        # self.X = tf.placeholder("float", [None, self.num_features])
        # self.Y = tf.placeholder("float", [None])
        self.input_ = tflearn.input_data(shape=[None, self.num_features], \
                                         data_preprocessing=data_prep)
        # self.linear = tflearn.single_unit(self.input_)
        # print self.linear
        self.linear = tflearn.fully_connected(self.input_, 1, bias=False)
        self.regression = tflearn.regression(self.linear, \
                                             optimizer='sgd', \
                                             loss='mean_square', \
                                             learning_rate=self.learning_rate, \
                                             batch_size=self.batch_size)
        self.m = tflearn.DNN(self.regression)
        """
        #[Note: Try 'optimal' for alpha]
        #[Note: Inside step function, don't ignore bias term]
        """
        self.regressor = linear_model.SGDRegressor(loss='squared_loss',\
                                                   penalty='l2',\
                                                   alpha = '0.3',\
                                                   fit_intercept = True,\
                                                   n_iter = 5,\
                                                   shuffle = True,\
                                                   verbose = 0,\
                                                   learning_rate = 'invscaling',\
                                                   eta0 = 0.01,\
                                                   power_t = 0.25,\
                                                   warm_start = True,\
                                                   average=True)
        """
        self.regressor = linear_model.SGDRegressor(alpha=0.5, warm_start=True)
        if visualize:
            self.t1.start()

    def learn(self, batch_learn = True):
        curr_episode = 0
        while curr_episode < self.total_episodes:
            planning_prob = self.sampleFromDatabase()
            graph = planning_prob[0]
            start_list = planning_prob[1]
            goal_list = planning_prob[2]
            # Initialize visualization
            if self.visualize:
                self.img = np.ones([graph.width, graph.height, 3]) * 255
                for start in start_list:
                    self.img[start[0], start[1]] = (0, 255, 0)
                for goal in goal_list:
                    self.img[goal[0], goal[1]] = (0, 0, 255)
                for i in graph.walls:
                    self.img[i[0], i[1]] = (0, 0, 0)

            w_mix = self.sampleMixture()
            # w_mix = self.weights_buffer[-1]
            w = self.learningBestFirstSearch(w_mix, planning_prob, visualize)
            self.weights_buffer.append(w)
            curr_episode += 1
            print "W", self.weights_buffer

    def sampleFromDatabase(self):
        idx = np.random.randint(0, len(self.env_database))
        print "*******************IDX*******************", idx
        return self.env_database[idx]

    def sampleMixture(self):
        # [TODO: Figure out best way to sample from an ML perspective]
        coin_toss = random.random()
        if coin_toss < self.lambda_factor or len(self.weights_buffer) == 1:
            return np.asarray(self.weights_buffer[0])
        else:
            idx = np.random.randint(1, len(self.weights_buffer))
            return np.asarray(self.weights_buffer[idx])

    def learningBestFirstSearch(self, w_mix, planning_prob, visualize=True):
        print("w_mix", w_mix)
        graph = planning_prob[0]
        start_list = planning_prob[1]
        goals_list = planning_prob[2]
        t = 0
        s = SearchAgent(graph, start_list, goals_list, self.base_heuristic, visualize)
        curr_weights = w_mix
        # Reset episode database before start of episode
        self.parent_database = []
        self.child_database = []
        self.feature_database = []
        self.best_cost_database = []
        self.e_dot_database = []
        print("Start New Episode")
        while t < self.episode_length:
            done, parent, child, feature_vec, best_cost, e_dot = s.step(curr_weights, self.base_heuristic)
            self.img[parent[0], parent[1]] = [255, 0, 0]
            if child is None or feature_vec is None:
                # print(parent, child, feature_vec, e_dot)
                continue
            # print "parent", parent, "child", child, "feature_vec", feature_vec, "cost", best_cost, "hp", self.base_heuristic(
                # parent, goals_list[0]), "hc", self.base_heuristic(child, goals_list[0]), "edot", e_dot

            t += 1
            if done:
                if self.include_terminal:
                    self.parent_database.append(parent)
                    self.child_database.append(child)
                    self.feature_database.append(feature_vec)
                    self.best_cost_database.append(best_cost)
                    self.e_dot_database.append(e_dot)
                print("Episode Finished")
                break
            else:
                self.parent_database.append(parent)
                self.child_database.append(child)
                self.feature_database.append(feature_vec)
                self.best_cost_database.append(best_cost)
                self.e_dot_database.append(e_dot)  # tflearn wants this

        print("Initiate learning")
        temp = []
        for i in self.e_dot_database:
            temp.append(float(i))
        self.e_dot_database = temp
        self.regressor.fit(np.array(self.feature_database), np.array(self.e_dot_database))

        print self.regressor.coef_
        return tuple(self.regressor.coef_)

    def display(self):
        cv2.namedWindow('Planning', cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow('Planning', self.img)
            cv2.waitKey(30)
