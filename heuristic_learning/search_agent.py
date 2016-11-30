#!/usr/bin/env python
import os
import sys
import cv2
from threading import Thread
import time
from math import fabs

sys.path.insert(0, os.path.abspath('..'))

from data_structures.PriorityQueue import *
from graphs.GridWithWeights import *
from graphs.HeuristicFunctions import *
from utils.planner_utils import *
import numpy as np


class SearchAgent(object):
    def __init__(self, graph, start_list, goal_list, base_heuristic, visualize=True):
        self.graph = graph
        self.start_list = start_list
        self.goal_list = goal_list
        self.base_heuristic = base_heuristic
        # self.w_curr = w_curr
        self.init = True
        self.FEATURE = []
        self.MAX_FEATURE = np.array([0.0])
        self.MIN_FEATURE = np.array([0.0])
        self.visualize = visualize
        self.frontier = PriorityQueue()
        self.came_from = {}
        self.cost_so_far = {}
        self.depth_so_far = {}
        self.obs_so_far = set()
        self.feature_means = []
        self.feature_std_de = []
        self.num_expansions = 0
        for start in self.start_list:
            self.frontier.put(start, 0)
            self.came_from[start] = None
            self.cost_so_far[start] = 0
            self.depth_so_far[start] = 0
            self.FEATURE.append(getNodeFeatures(start, goal_list[0], base_heuristic, 0, self.obs_so_far, 0))
            # print "FEATURE", self.FEATURE
            # print(self.start_list, self.goal_list, self.graph.walls)

    # def setWeights(self, w):
    # 	self.w_curr = w

    def step(self, weights, featureFunc):
        done = False
        if self.frontier.empty():
            done = True
            print("Done coz front empty")
            return done, None, None, None, None, None
        else:
            current, curr_priority = self.frontier.get()
            # print current

            # print(current)
            if current in self.goal_list:
                done = True
                # path_so_far = reconstruct_path(self.came_from, self.start_list[0], current, self.cost_so_far)
                print("Done coz found goal")
                goal_feature_vec = getEdgeFeatures(current, current, self.goal_list, Euclidean, self.cost_so_far,
                                                   self.obs_so_far, self.depth_so_far)
                self.FEATURE.append(goal_feature_vec)

                if self.init:
                    temp = zip(*self.FEATURE)
                    for i in range(1):
                        self.MAX_FEATURE[i] = max(temp[i]) + 0.001
                        self.MIN_FEATURE[i] = min(temp[i])
                    self.init = False
                else:
                    for i in range(1):
                        if goal_feature_vec[i] > self.MAX_FEATURE[i]:
                            self.MAX_FEATURE[i] = goal_feature_vec[i]
                        elif goal_feature_vec[i] < self.MIN_FEATURE[i]:
                            self.MIN_FEATURE[i] = goal_feature_vec[i]
                goal_feature_vec = normalize(goal_feature_vec, self.MAX_FEATURE, self.MIN_FEATURE)
                return done, current, current, goal_feature_vec, self.cost_so_far[current], 0
            # Path to current node popped is path_so_far
            # path_so_far = reconstruct_path(self.came_from, self.start_list[0], current, self.cost_so_far)
            neighbors, obs_neighbors = self.graph.neighbors(current)
            # Add obstacle neighbors to obs_so_far
            for obs_neighbor in obs_neighbors:
                self.obs_so_far.add(obs_neighbor)
            best_c_plus_h = float("inf")
            best_neighbor = ()
            best_feature_vec = np.array([0])
            best_cost = 0
            best_e_dot = 0
            for next in neighbors:
                # Get edge features
                feature_vec = getEdgeFeatures(current, next, self.goal_list, Euclidean, self.cost_so_far,
                                              self.obs_so_far, self.depth_so_far)
                self.FEATURE.append(feature_vec)

                if self.init:
                    temp = zip(*self.FEATURE)
                    for i in range(1):
                        self.MAX_FEATURE[i] = max(temp[i]) + 0.001
                        self.MIN_FEATURE[i] = min(temp[i])
                    self.init = False
                else:
                    for i in range(1):
                        if feature_vec[i] > self.MAX_FEATURE[i]:
                            self.MAX_FEATURE[i] = feature_vec[i]
                        elif feature_vec[i] < self.MIN_FEATURE[i]:
                            self.MIN_FEATURE[i] = feature_vec[i]
                feature_vec = normalize(feature_vec, self.MAX_FEATURE, self.MIN_FEATURE)
                new_cost = self.cost_so_far[current] + self.graph.cost(current, next)
                new_depth = self.depth_so_far[current] + 1
                if next not in self.cost_so_far or new_cost < self.cost_so_far[next]:
                    succ_priority = SearchAgent.getHcapSucc(next, feature_vec, weights, self.base_heuristic, self.goal_list)
                    # c_plus_h = SearchAgent.getCPlusH(self.graph.cost(current,next), next, feature_vec, weights, self.base_heuristic, self.goal_list)
                    c_plus_h = self.graph.cost(current, next) + succ_priority
                    if c_plus_h < best_c_plus_h:
                        best_c_plus_h = c_plus_h
                        best_neighbor = next
                        best_feature_vec = feature_vec
                        best_cost = self.graph.cost(current, best_neighbor)
                        best_e_dot = SearchAgent.getEDot(best_cost, curr_priority, succ_priority)
                    self.cost_so_far[next] = new_cost
                    self.depth_so_far[next] = new_depth
                    # Get priority using current error rate estimate-
                    self.frontier.put(next, succ_priority)
                    self.came_from[next] = current
            time.sleep(0.01)
            # print("Best Feature", best_feature_vec)
            if best_neighbor:
                return done, current, best_neighbor, best_feature_vec, best_cost, best_e_dot
            else:
                return done, current, None, None, None, None

    @staticmethod
    def getEDot(cost, parent_priority, succ_priority):
        """From Equation: e_dot = (c(x,x') - r_cap)/r_cap
            r_cap = h_cap(x) - h_cap(x')
        """
        # h_cap_parent = SearchAgent.getHcapParent(parent, feature_vec, weights, base_heuristic, goal_list)
        # h_cap_child = SearchAgent.getHcapSucc(child, feature_vec, weights, base_heuristic, goal_list)
        # h_cap_parent = base_heuristic(parent, goal_list[0])
        # h_cap_child= base_heuristic(child, goal_list[0])
        # r_cap = h_cap_parent - h_cap_child
        # if fabs(r_cap < 0.0000
        # e_dot = (cost - r_cap)/r_cap
        e_dot = (cost + succ_priority - parent_priority) / (succ_priority - parent_priority)
        return e_dot

    @staticmethod
    def getHcapParent(node, feature_vec, weights, base_heuristic, goal_list):
        # From Equation: h_cap = h_base*(e_cap_dot + 1)
        h_base = 0
        for goal in goal_list:  # This will work for multiple goals as well
            h_base += base_heuristic(node, goal)
        e_cap_dot = SearchAgent.getEcapDot(feature_vec, weights)
        h_cap = h_base * (1 + e_cap_dot)
        return h_cap

    @staticmethod
    def getHcapSucc(node, feature_vec, weights, base_heuristic, goal_list):
        # From Equation: h_cap = h_base*(1 - e_cap_dot)
        h_base = 0
        for goal in goal_list:  # This will work for multiple goals as well
            h_base += base_heuristic(node, goal)
        e_cap_dot = SearchAgent.getEcapDot(feature_vec, weights)
        h_cap = h_base * (1 - e_cap_dot)
        return h_cap

    @staticmethod
    def getEcapDot(feature_vec, weights):
        # From Equation: e_cap_dot = w*features
        return np.dot(feature_vec, weights)

        # @staticmethod
        # def getCPlusH(cost, next, feature_vec, weights, base_heuristic, goal_list):
        # 	#From Equation: h(s) <= c(s,s') + h(s')
        # 	return cost + SearchAgent.getHcapSucc(next, feature_vec, weights, base_heuristic, goal_list)
