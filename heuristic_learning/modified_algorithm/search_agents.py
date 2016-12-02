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
from graphs.HeuristicFunctions import *
from utils.planner_utils import *
import numpy as np
import operator


class SearchAgent(object):
    def __init__(self):

    @staticmethod
    def getEDot(cost, h_x, h_x_dash):
        """From Equation: e_dot = (c(x,x') - r_cap)/r_cap
            r_cap = h_cap(x) - h_cap(x')
        """
        e_dot = (cost + h_x_dash - h_x) / (h_x - h_x_dash)
        return e_dot

    @staticmethod
    def getHcap(node, feature_vec, weights, bias, h_base):
        # From Equation: h_cap = h_base*(1 + e_cap_dot)
        e_cap_dot = SearchAgent.getEcapDot(feature_vec, np.array(weights))
        h_cap = h_base * (1 + e_cap_dot)
        return h_cap

    @staticmethod
    def getEcapDot(feature_vec, weights, bias):
        # From Equation: e_cap_dot = w*features
        return np.dot(feature_vec, weights) + bias

class BatchSearchAgent(SearchAgent):
    def __init__(self, graph, start, goal, base_heuristic, feature_map):
        self.graph  = graph
        self.frontier = PriorityQueue()
        self.start = start
        self.goal = goal
        self.base_heuristic = base_heuristic
        self.feature_map = feature_map
        self.frontier.put(start, 0 + self.base_heuristic(start, goal))
        self.came_from = {}
        self.cost_so_far = {}

    def step(self):
    #Does an A-star Search with base_heuristic and reutrns errors observed + features observed
    done = False
    if self.frontier.empty():
        done = True
        print("Done coz front empty")
        return done, None, None, None
    else:
        current, curr_priority = self.frontier.get()
        h_x = self.base_heuristic(current, goal)
        # print(current)
        if current == self.goal:
            done = True
            print("Done coz found goal")
            return done, goal, 0, 0
        neighbors, obs_neighbors = self.graph.neighbors(current)
        #Add obstacle neighbors to obs_so_far
        # for obs_neighbor in obs_neighbors:
        #     self.obs_so_far.add(obs_neighbor) 
        
        best_c_plus_h = float("inf")
        best_feature_vec = []
        best_error = 0

        for next in neighbors:
            
            edge_cost = self.graph.cost(current, next)
            new_cost = cost_so_far[current] + edge_cost
            
            
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                new_depth = depth_so_far[current] + 1
                h_x_dash = base_heuristic(next, goal) 
                #Check if it is best child
                c_plus_h =  edge_cost + h_x_dash
                priority = new_cost + h_x_dash
                if c_plus_h < best_c_plus_h:
                    best_c_plus_h = c_plus_h
                    best_feature_vec = self.feature_map[next]
                    best_error = SearchAgent.getEDot(edge_cost, h_x, h_x_dash)
                cost_so_far[next] = new_cost
                frontier.put(next, priority)
                came_from[next] = current
        
        return done, current, best_error, best_feature_vec
        

class OnlineSearchAgent(SearchAgent):
    def __init__(self, graph, start, goal, base_heuristic, feature_map, a_star = True):
        self.graph  = graph
        self.frontier = PriorityQueue()
        self.start = start
        self.goal = goal
        self.base_heuristic = base_heuristic
        self.feature_map = feature_map
        self.frontier.put(start, 0 + self.base_heuristic(start, goal))
        self.came_from = {}
        self.cost_so_far = {}
        self.a_star = a_star
        self.regressor = linear_model.SGDRegressor(alpha=0.5, warm_start = True)

    def step(self):
        done = False
        if self.frontier.empty():
            print("Done coz frontier empty")
            done = True
            return done, None
        else:
            current, curr_priority = self.frontier.get()
            #Calculate h_x
            h_x = self.base_heuristic(current, goal)
            if current == goal:
                print("Done coz goal found")
                done = True
                return done, current
            neighbors, obs_neighbors = self.graph.getNeighbors(current)
            #We need to update the weights now
            #Calculate best child according to the consistency equation x* = argmin(c(x,x') + h(x'))
            best_c_plus_h_x_dash, best_child_idx = min(enumerate([self.graph.cost(current, i) + self.base_heuristic(i, self.goal)]), key=operator.itemgetter(1))
            best_child = neighbors[best_child_idx]
            best_child_features = self.feature_map[best_child]
            best_h_x_dash = self.base_heuristic(best_child, goal)
            #Calculate error due to base heuristic on best_child
            best_error = SearchAgent.getEDot(self.graph.cost(current, best_child), h_x, best_h_x_dash)
            self.regressor.fit(best_child_features, best_error)
            curr_weights = self.regressor.coef_
            curr_bias = self.intercept_
            
            #Now do the typical A-star thing
            for next in neighbors:
                edge_cost = self.graph.cost(current, next)
                new_cost = self.cost_so_far[current] + edge_cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:               
                    new_depth = self.depth_so_far[current] + 1
                    new_feature_vec = self.feature_map[next]
                    h_x_dash_cap = SearchAgent.getHcapDash(next, new_feature_vec, curr_weights, curr_bias, self.base_heuristic(next, goal))
                    if self.a_star:
                        next_priority = new_cost + h_x_dash_cap
                    else:
                        next_priority = h_x_dash_cap
                    self.frontier.put(next, next_priority)
                    self.cost_so_far[next] = new_cost
                    self.came_from[next] = current
            return done, current




def TestAgent(SearchAgent):
    def __init__(self, graph, start, goal, base_heuristic, feature_map):
        self.graph  = graph
        self.frontier = PriorityQueue()
        self.start = start
        self.goal = goal
        self.base_heuristic = base_heuristic
        self.feature_map = feature_map
        self.came_from = {}
        self.cost_so_far = {}
        
        self.frontier.put(start, 0 + self.base_heuristic(start, goal))
       

    def run_test(self, weights, bias):
    num_expansions = 0
    best_error_buffer = []
    while not self.frontier.empty():
        current, current_priority = self.frontier.get()
        num_expansions += 1
        h_x = self.base_heuristic(current, self.goal)
        if current == goal:
            break

        neighbors, obs_neighbors = self.graph.neighbors(current)
        #Add obstacle neighbors to obs_so_far
        # for obs_neighbor in obs_neighbors:
        #     self.obs_so_far.add(obs_neighbor) 

        best_c_plus_h = float("inf")
        best_feature_vec = []
        best_error = 0
        for next in neighbors:
            
            edge_cost = self.graph.cost(current, next)
            new_cost = cost_so_far[current] + edge_cost
            new_depth = depth_so_far[current] + 1
            h_x_dash = base_heuristic(next, goal) 
            
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                #Check if it is best child
                c_plus_h =  edge_cost + h_x_dash_cap #Now we check errors with improved heuristics
                feature_vec = self.feature_map[next]
                h_x_dash_cap = SearchAgent.get(next, feature_vec, weights, h_x_dash)
                priority = new_cost + h_x_dash_cap
                if c_plus_h < best_c_plus_h:
                    best_c_plus_h = c_plus_h
                    best_feature_vec = feature_map[(current, next)]
                    best_error = SearchAgent.getEDot(edge_cost, h_x, h_x_dash)

                cost_so_far[next] = new_cost
                frontier.put(next, priority)
                came_from[next] = current
            best_error_buffer.append(best_error)

    return num_expansions, best_error_buffer
