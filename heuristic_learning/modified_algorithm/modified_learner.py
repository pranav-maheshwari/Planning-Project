#!/usr/bin/env python
import os
import sys
import cv2
from threading import Thread

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import random

from search_agents import BatchSearchAgent, OnlineSearchAgent, TestAgent
from utils.planner_utils import *
from sklearn import linear_model

("Import done")


class Learner:
    def __init__(self,\
                total_episodes,\
                episode_length,\
                learning_rate,\
                test_env_database,\
                seed,\
                base_heuristic,\
                include_terminal = False,\
                visualize = True):

        self.learning_rate = learning_rate
        self.episode_length = episode_length
        self.base_heuristic = base_heuristic
        self.include_terminal = include_terminal
        self.visualize = visualize
        self.img = np.array([0])
        self.t1 = Thread(target=self.display)
        self.test_env_database = test_env_database
        np.random.seed(seed)
    
        #[Note: Try 'optimal' for alpha]
        #[Note: Inside step function, don't ignore bias term] 
        if self.visualize:
            self.t1.start()

    def learn_batch_mode(self, max_env_num = 100, test = True):
        #In batch mode we learn on a single environment and 
        # curr_env = 100
        for curr_env in xrange(max_env_num):
            learned_env_weights = dict()
            # print len(self.test_env_database)
            # for curr_env in xrange(len(self.test_env_database)):
            planning_prob = self.test_env_database[curr_env]
            # Initialize visualization
            if self.visualize:
                self.initialize_image(planning_prob)        
            w_b = self.learningBestFirstSearchBatch(planning_prob)
            #Store learned weights in dictionary for environments
            learned_env_weights[curr_env] = w_b
            #Now we test weights in an environment
            if test:
                # num_expansions, errors = self.test_weights_in_env(w_b, planning_prob) #[0.0644, -0.0297, 0.0255, -0.0643, 0.0154], 0)
                #[-0.01555318, -0.35625475, -0.55566625, -0.5029324]), -0.41105624)
                num_expansions, errors = self.test_weights_in_env(w_b, planning_prob)
                print "Num of expansions during test(batch): ", num_expansions

        return learned_env_weights

    def learn_online_mode(self, max_env_num = 100):
        for curr_env in xrange(max_env_num):
            # for curr_env in xrange(len(self.test_env_database)) :
            planning_prob = self.test_env_database[curr_env]
            # Initialize visualization
            if self.visualize:
                self.initialize_image(planning_prob)
            self.learningBestFirstSearchOnline(planning_prob)


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
            done, current, error_target, feature_vec = s.step()
            self.img[current[0], current[1]] = [255, 0, 0]
            if current is None or error_target is None or feature_vec is None:
                # print(parent, child, feature_vec, e_dot)
                continue
            # print "parent", parent, "child", child, "feature_vec", feature_vec, "cost", best_cost, "hp", self.base_heuristic(
                # parent, goals_list[0]), "hc", self.base_heuristic(child, goals_list[0]), "edot", e_dot
            t += 1
            if done:
                if self.include_terminal:
                    feature_database.append(feature_vec)
                    error_database.append(e_dot)
                print("Episode Finished")
                break
            else:
                feature_database.append(feature_vec)
                error_database.append(error_target)  
        print("Num Expansions A*", t)
        print("Initiate learning")
        regressor = linear_model.SGDRegressor(alpha = 0.003, verbose = 0, n_iter = 5, fit_intercept = True)#, average=True)
        # regressor = linear_model.LinearRegression()

        # print error_database
        # print feature_database
        try:
            check = len(feature_database[0])
        except TypeError:
            feature_database = [[i] for i  in feature_database]
        print feature_database
        # print np.asarray(error_database).shape
        # print np.asarray(feature_database).shape
        regressor.fit(feature_database, error_database)# coef_init= [0]*len(feature_database[0]))
        # print regressor.coef_
        # print regressor.intercept_
        return (regressor.coef_, regressor.intercept_) #[NOTE: Take bias terms into account as well]

    def test_weights_in_env(self, w_b, planning_prob):
        if self.visualize:
            self.initialize_image(planning_prob) 
        graph = planning_prob[0]
        start = planning_prob[1][0]
        goal = planning_prob[2][0]
        feature_map = planning_prob[3]
        weights = w_b[0]
        bias = w_b[1]
        # temp = sum(weights) + bias
        # bias = (1.0*bias)/temp
        # bias= bias*len(weights)
        # weights = [(i*1.0)/temp for i in weights]
        # print weights, bias
        test_agent = TestAgent(graph, start, goal, self.base_heuristic, feature_map)
        t = 0
        # feature_database = []
        error_database = []
        print("Start Testing Episode")
        while t < self.episode_length:
            done, current, error_target = test_agent.run_test(weights, bias)
            t += 1
            self.img[current[0], current[1]] = [255, 0, 0]
            if current is None or error_target is None:
                continue
            if done:
                if self.include_terminal:
                    error_database.append(e_dot)
                print("Episode Finished")
                break
            else:  
                error_database.append(error_target)  

        num_expansions = t
        return num_expansions, error_database

    def learningBestFirstSearchOnline(self, planning_prob):

        graph = planning_prob[0]
        start_list = planning_prob[1]
        goals_list = planning_prob[2]
        feature_map = planning_prob[3]
        t = 0
        s = OnlineSearchAgent(graph, start_list[0], goals_list[0], self.base_heuristic, feature_map, True)
        # Reset episode database before start of episode
        print("Start New Episode")
        while t < self.episode_length:
            done, current = s.step()
            if current is None:
                # print(parent, child, feature_vec, e_dot)
                continue
            self.img[current[0], current[1]] = [255, 0, 0]
            t += 1
            if done:
                print("Episode Finished")
                break 
        print "Num expansions ", t

    def display(self):
        cv2.namedWindow('Planning', cv2.WINDOW_NORMAL)
        while True:
            cv2.imshow('Planning', self.img)
            cv2.waitKey(30)

    def initialize_image(self, planning_prob):
        graph = planning_prob[0]
        start_list = planning_prob[1]
        goal_list = planning_prob[2]
        
        self.img = np.ones([graph.width, graph.height, 3]) * 255
        for start in start_list:
            self.img[start[0], start[1]] = (0, 255, 0)
        for goal in goal_list:
            self.img[goal[0], goal[1]] = (0, 0, 255)
        for i in graph.walls:
            self.img[i[0], i[1]] = (0, 0, 0)
