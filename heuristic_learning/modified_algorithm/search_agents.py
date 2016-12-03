#!/usr/bin/env python
import os
import sys
import cv2
from threading import Thread
import time
from math import fabs

sys.path.insert(0, os.path.abspath('../..'))

from data_structures.PriorityQueue import *
from graphs.GridWithWeights import *
from graphs.GridWithWeightsSoftObs import *
from graphs.HeuristicFunctions import *
from utils.planner_utils import *
import numpy as np
import operator
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn import linear_model 


class SearchAgent(object):
    @staticmethod
    def getEDot(cost, h_x, h_x_dash):
        """From Equation: e_dot = (c(x,x') - r_cap)/r_cap
            r_cap = h_cap(x) - h_cap(x')
        """
        e_dot = float(cost + h_x_dash - h_x)/ (h_x - h_x_dash)
        return e_dot

    @staticmethod
    def getHcap(node, feature_vec, weights, bias, h_x_base, h_x_dash_base, cost):
        # From Equation: h_cap = h_base*(1 + e_cap_dot)
        e_cap_dot = SearchAgent.getEcapDot(feature_vec, np.array(weights), bias)
        h_cap = h_x_dash_base - e_cap_dot*(h_x_base - h_x_dash_base)
        # h_cap = h_x_base - cost/(-1 - e_cap_dot)
        return h_cap

    @staticmethod
    def getEcapDot(feature_vec, weights, bias):
        # From Equation: e_cap_dot = w*features
        return np.dot(feature_vec, weights) + bias

class BatchSearchAgent(SearchAgent):
    def __init__(self, graph, start, goal, base_heuristic, feature_map, a_star = True):
        self.graph  = graph
        self.frontier = PriorityQueue()
        self.start = start
        self.goal = goal
        self.base_heuristic = base_heuristic
        self.feature_map = feature_map
        self.frontier.put(start, 0 + self.base_heuristic(start, goal))
        self.a_star = a_star
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[start] = None
        self.cost_so_far[start] = 0
        # self.depth_so_far[start] = 0

    def step(self):
        #Does an A-star Search with base_heuristic and reutrns errors observed + features observed
        done = False
        if self.frontier.empty():
            done = True
            print("Done coz front empty")
            return done, None, None, None
        else:
            current, curr_priority = self.frontier.get()
            h_x = self.base_heuristic(current, self.goal)
            # print curr_priority
            if current == self.goal:
                done = True
                print("Done coz found goal")
                return done, current, 0, 0
            neighbors = self.graph.neighbors(current)
                
            best_c_plus_h = float("inf")
            best_feature_vec = None
            best_error = None
            for next in neighbors:
                edge_cost = self.graph.cost(current, next)
                new_cost = self.cost_so_far[current] + edge_cost
                if next not in self.cost_so_far or new_cost < self.cost_so_far[next]:
                    # new_depth = depth_so_far[current] + 1
                    h_x_dash = self.base_heuristic(next, self.goal) 
                    #Check if it is best child
                    c_plus_h =  edge_cost + h_x_dash

                    if self.a_star:
                        priority = new_cost + h_x_dash
                    else:
                        priority = h_x_dash
                    if c_plus_h < best_c_plus_h:
                        best_c_plus_h = c_plus_h
                        # best_feature_vec = np.array([min(self.feature_map[next]), h_x_dash/128.0])
                        best_feature_vec = min(self.feature_map[next])
                        best_error = SearchAgent.getEDot(edge_cost, h_x, h_x_dash)
                        
                        # print edge_cost + h_x_dash - h_x
                    
                    self.cost_so_far[next] = new_cost
                    self.frontier.put(next, priority)
                    self.came_from[next] = current
                time.sleep(0.01)
            return done, current, best_error, best_feature_vec
        

class OnlineSearchAgent(SearchAgent):
    def __init__(self, graph, start, goal, base_heuristic, feature_map, a_star = True):
        self.graph  = graph
        self.frontier = PriorityQueue()
        self.start = start
        self.goal = goal
        self.base_heuristic = base_heuristic
        self.feature_map = feature_map
        self.frontier.put(start, 0 + self.base_heuristic(start, self.goal))
        self.came_from = {}
        self.cost_so_far = {}
        self.a_star = a_star
        self.regressor = linear_model.SGDRegressor(alpha=0.03, verbose = 0, n_iter = 5, fit_intercept = True, warm_start = True)
        self.came_from[start] = None
        self.cost_so_far[start] = 0
        # self.depth_so_far[start] = 0


    def step(self):
        done = False
        if self.frontier.empty():
            print("Done coz frontier empty")
            done = True
            return done, None
        else:
            current, curr_priority = self.frontier.get()
            #Calculate h_x
            h_x = self.base_heuristic(current, self.goal)
            if current == self.goal:
                print("Done coz goal found")
                done = True
                return done, current
            neighbors = self.graph.neighbors(current)
            #We need to update the weights now
            #Calculate best child according to the consistency equation x* = argmin(c(x,x') + h(x'))
            # best_child_idx, best_c_plus_h_x_dash = min(enumerate([self.graph.cost(current, i) + self.base_heuristic(i, self.goal)] for i in neighbors), key=operator.itemgetter(1))
            # print best_child_idx
            # print best_c_plus_h_x_dash
            # best_child = neighbors[best_child_idx]
            # best_child_features = self.feature_map[best_child]
            # best_h_x_dash = self.base_heuristic(best_child, self.goal)
            # #Calculate error due to base heuristic on best_child
            # best_error = SearchAgent.getEDot(self.graph.cost(current, best_child), h_x, best_h_x_dash)
            # if best_error != 0:
            #     print "BEST", best_error
            best_c_plus_h = float("inf")
            best_feature_vec = None
            best_error = None
            for next in neighbors:
                edge_cost = self.graph.cost(current, next)
                new_cost = self.cost_so_far[current] + edge_cost
                if next not in self.cost_so_far or new_cost < self.cost_so_far[next]:
                    # new_depth = depth_so_far[current] + 1
                    h_x_dash = self.base_heuristic(next, self.goal) 
                    #Check if it is best child
                    c_plus_h =  edge_cost + h_x_dash
                    if c_plus_h < best_c_plus_h:
                        best_c_plus_h = c_plus_h
                        # best_feature_vec = np.array([min(self.feature_map[next]), h_x_dash/128.0])
                        best_feature_vec = min(self.feature_map[next])
                        best_error = SearchAgent.getEDot(edge_cost, h_x, h_x_dash)
            # try:
            #     check = len(best_child_features[0])
            # except TypeError:
            #     best_child_features = [[i] for i  in best_child_features]
            #.reshape(len(best_child_features), 1)
            if best_error is not None and best_feature_vec is not None:
                self.regressor.fit(best_feature_vec, [best_error])
            
            curr_weights = self.regressor.coef_
            curr_bias = self.regressor.intercept_
            print curr_weights, curr_bias

            
            #Now do the typical A-star thing
            for next in neighbors:
                edge_cost = self.graph.cost(current, next)
                new_cost = self.cost_so_far[current] + edge_cost
                if next not in self.cost_so_far or new_cost < self.cost_so_far[next]:               
                    # new_depth = self.depth_so_far[current] + 1
                    # new_feature_vec = np.array([min(self.feature_map[next]), h_x_dash/128.0])
                    new_feature_vec = min(self.feature_map[next])
                    h_x_dash_cap = SearchAgent.getHcap(next, new_feature_vec, curr_weights, curr_bias, h_x, self.base_heuristic(next, self.goal), edge_cost)
                    if self.a_star:
                        next_priority = new_cost + h_x_dash_cap
                    else:
                        next_priority = h_x_dash_cap
                    
                    self.frontier.put(next, next_priority)
                    self.cost_so_far[next] = new_cost
                    self.came_from[next] = current
            time.sleep(0.1)
            return done, current




class TestAgent(SearchAgent):

    def __init__(self, graph, start, goal, base_heuristic, feature_map, a_star = True):
        self.graph  = graph
        self.frontier = PriorityQueue()
        self.start = start
        self.goal = goal
        self.base_heuristic = base_heuristic
        self.feature_map = feature_map
        self.came_from = {}
        self.cost_so_far = {}
        self.a_star = a_star
        self.frontier.put(start, 0 + self.base_heuristic(start, goal))
        self.cost_so_far[start] = 0
        self.came_from[start] = None
       

    def run_test(self, weights, bias):
        done = False
        if self.frontier.empty():
            done = True
            print("Done coz front empty")
            return done, None, None
        else:
            current, curr_priority = self.frontier.get()
            h_x = self.base_heuristic(current, self.goal)
            # print(current)
            if current == self.goal:
                done = True
                print("Done coz found goal")
                return done, current, 0
            
            neighbors = self.graph.neighbors(current)
            best_c_plus_h = float("inf")
            best_feature_vec = None
            best_error = None
            for next in neighbors:
                
                edge_cost = self.graph.cost(current, next)
                new_cost = self.cost_so_far[current] + edge_cost
                # new_depth = self.depth_so_far[current] + 1
                h_x_dash = self.base_heuristic(next, self.goal) 
                
                if next not in self.cost_so_far or new_cost < self.cost_so_far[next]:
                    #Check if it is best child
                    c_plus_h =  edge_cost + h_x_dash #Now we check errors with improved heuristics
                    # feature_vec = np.array([min(self.feature_map[next]), h_x_dash/128.0])
                    feature_vec = min(self.feature_map[next])
                    h_x_dash_cap = SearchAgent.getHcap(next, feature_vec, weights, bias, h_x, h_x_dash, edge_cost)
                    if self.a_star:
                        priority = new_cost + h_x_dash_cap
                    else:
                        priority = h_x_dash_cap
                    # print h_x_dash_cap, h_x_dash, feature_vec
                    if c_plus_h < best_c_plus_h:
                        best_c_plus_h = c_plus_h
                        # best_feature_vec = np.array([min(self.feature_map[next]), h_x_dash/128.0])
                        best_feature_vec = min(self.feature_map[next])
                        best_error = SearchAgent.getEDot(edge_cost, h_x, h_x_dash)

                    self.cost_so_far[next] = new_cost
                    self.frontier.put(next, priority)
                    self.came_from[next] = current
                time.sleep(0.01)
            return done, current, best_error
