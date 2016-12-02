#!/usr/bin/env python
import os
import sys
import cv2
from threading import Thread

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import random

from search_agents import BatchSearchAgent, OnlineSearchAgent
from utils.planner_utils import *
from sklearn import linear_model

print("Import done")


class Learner:
    def __init__(total_episodes, \
            episode_length, \
            learning_rate, \
            batch_size, \
            test_env_database, \
            seed = 1234, \
            base_heuristic, \
            include_terminal = False, \
            batch_train, \
            visualize = True):

        self.total_episodes = total_episodes
        self.learning_rate = learning_rate
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.base_heuristic = base_heuristic
        self.include_terminal = include_terminal
        self.visualize = visualize
        self.img = np.array([0])
        self.t1 = Thread(target=self.display)
        self.test_env_database = test_env_database
        np.random.seed(seed)
        self.learned_env_weights_batch = dict()

        
        #[Note: Try 'optimal' for alpha]
        #[Note: Inside step function, don't ignore bias term]

        self.regressor = linear_model.SGDRegressor(alpha=0.5)
        if self.visualize:
            self.t1.start()

    def learn_batch_mode(self, test = True):
        #In batch mode we learn on a single environment and 
        curr_env = 0

        for curr_env in xrange(len(self.test_env_database) :
            planning_prob = self.test_env_database[curr_episode]
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

            w = self.learningBestFirstSearchBatch(planning_prob, self.base_heuristic)#, self.feature_map)
            #Store learned weights in dictionary for environments
            self.learned_env_weights_batch[curr_episode] = w
            #Now we test weights in an environment
            if test:
                num_expansions, errors = self.test_weights_in_env(w, planning_prob)


    def learningBestFirstSearchBatch(self, planning_prob):

        graph = planning_prob[0]
        start_list = planning_prob[1]
        goals_list = planning_prob[2]
        feature_map = planning_prob[3]
        t = 0
         #Compute Feature Map from Pranav's environment here
        s = BatchSearchAgent(graph, start_list[0], goals_list[0], self.base_heuristic, feature_map)
        # Reset episode database before start of episode
        feature_database = []
        error_database = []

        print("Start New Episode")
        while t < self.episode_length:
            done, error_target, feature_vec = s.step(curr_weights, self.base_heuristic)
            self.img[parent[0], parent[1]] = [255, 0, 0]
            if child is None or feature_vec is None:
                # print(parent, child, feature_vec, e_dot)
                continue
            # print "parent", parent, "child", child, "feature_vec", feature_vec, "cost", best_cost, "hp", self.base_heuristic(
                # parent, goals_list[0]), "hc", self.base_heuristic(child, goals_list[0]), "edot", e_dot
            t += 1
            if done:
                if self.include_terminal:
                    feature_database.append(feature_vec)
                    e_dot_database.append(e_dot)
                print("Episode Finished")
                break
            else:
                feature_database.append(feature_vec)
                e_dot_database.append(error_target)  
        print("Initiate learning")

        temp = []
        for i in self.e_dot_database:
            temp.append(float(i))
        self.e_dot_database = temp
        self.regressor.fit(np.array(self.feature_database), np.array(self.e_dot_database))

        print self.regressor.coef_
        return tuple(self.regressor.coef_) #[NOTE: Take bias terms into account as well]

    
    def test_weights_in_env(self, weights, planning_prob):
        graph = planning_prob[0]
        start = planning_prob[1][0]
        goal = planning_prob[2][0]
        feature_map = planning_prob[3]
        test_agent = TestAgent(graph, start, goal, self.base_heuristic, feature_map)
        num_expansions, errors = test_agent.run_test(weights)

    def learningBestFirstSearchOnline(self, planning_prob):

        graph = planning_prob[0]
        start_list = planning_prob[1]
        goals_list = planning_prob[2]
        feature_map = planning_prob[3]
        t = 0
        s = OnlineSearchAgent(graph, start_list[0], goals_list[0], self.base_heuristic, feature_map)
        # Reset episode database before start of episode
        print("Start New Episode")
        while t < self.episode_length:
            done, current = s.step(curr_weights, self.base_heuristic)
            if current is None:
                # print(parent, child, feature_vec, e_dot)
                continue
            self.img[current[0], current[1]] = [255, 0, 0]
            t += 1
            if done:
                print("Episode Finished")
                break 
        print("Initiate learning")

    def display(self):
        cv2.namedWindow('Planning', cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow('Planning', self.img)
            cv2.waitKey(30)
