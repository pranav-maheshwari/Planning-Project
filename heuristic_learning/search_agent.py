#!/usr/bin/env python
import os
import sys
import cv2
from threading import Thread
import time
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
		self.visualize = visualize
		self.frontier = PriorityQueue()
		self.came_from = {}
		self.cost_so_far = {}
		self.depth_so_far = {}
		self.obs_so_far = set()
		self.img = np.array([0])
		self.t1 = Thread(target = self.display)
		self.num_expansions = 0
		for start in self.start_list:
			self.frontier.put(start,0)
			self.came_from[start] = None
			self.cost_so_far[start] = 0
			self.depth_so_far[start] = 0
		if visualize:
			self.img = np.ones([graph.width, graph.height, 3])*255
			for start in self.start_list:
				self.img[start[0], start[1]] = (0,255,0)
			for goal in self.goal_list:
				self.img[goal[0], goal[1]] = (0,0,255)
			for i in self.graph.walls:
				self.img[i[0], i[1]] = (0, 0, 0)
			self.t1.start()
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
			current = self.frontier.get()
			# print current
			self.img[current[0], current[1]] = [255, 0, 0]
			# print(current)
			if current in self.goal_list:
				done = True
				# path_so_far = reconstruct_path(self.came_from, self.start_list[0], current, self.cost_so_far)
				print("Done coz found goal")
				goal_feature_vec = getEdgeFeatures(current, current, self.goal_list, Euclidean, self.cost_so_far, self.obs_so_far, self.depth_so_far)
				return done, current, current, goal_feature_vec, self.cost_so_far[current], 0
			#Path to current node popped is path_so_far
			# path_so_far = reconstruct_path(self.came_from, self.start_list[0], current, self.cost_so_far)
			neighbors, obs_neighbors = self.graph.neighbors(current)
			#Add obstacle neighbors to obs_so_far
			for obs_neighbor in obs_neighbors:
				self.obs_so_far.add(obs_neighbor)
			best_c_plus_h = float("inf")
			best_neighbor = ()
			best_feature_vec = np.array([0])
			best_cost = 0
			best_e_dot = 0
			for next in neighbors:
				#Get edge features
				feature_vec = getEdgeFeatures(current, next, self.goal_list, Euclidean, self.cost_so_far, self.obs_so_far, self.depth_so_far)
				# print feature_vec
				new_cost = self.cost_so_far[current] + self.graph.cost(current, next)
				new_depth = self.depth_so_far[current] + 1
				c_plus_h = SearchAgent.getCPlusH(new_cost, next, feature_vec, weights, self.base_heuristic, self.goal_list)
				if next not in self.cost_so_far or new_cost < self.cost_so_far[next]:
					if c_plus_h < best_c_plus_h:
						best_c_plus_h = c_plus_h
						best_neighbor = next
						best_feature_vec = feature_vec
						best_cost = new_cost
						best_e_dot = SearchAgent.getEDot(current, best_neighbor, best_cost, best_feature_vec, weights, self.base_heuristic, self.goal_list)
					self.cost_so_far[next] = new_cost
					self.depth_so_far[next] = new_depth
					#Get priority using current error rate estimate
					priority = SearchAgent.getHcap(next, feature_vec, weights, self.base_heuristic, self.goal_list)
					self.frontier.put(next, priority)
					self.came_from[next] = current
			time.sleep(0.5)
			# print("Best Feature", best_feature_vec)
			if best_neighbor:
				return done, current, best_neighbor, best_feature_vec, best_cost, best_e_dot
			else: 
				return done, current, None, None, None, None

	@staticmethod
	def getEDot(parent, child, cost, feature_vec, weights, base_heuristic, goal_list):
		"""From Equation: e_dot = (c(x,x') - r_cap)/r_cap
			r_cap = h_cap(x) - h_cap(x')
		"""
		h_cap_parent = SearchAgent.getHcap(parent, feature_vec, weights, base_heuristic, goal_list)
		h_cap_child = SearchAgent.getHcap(child, feature_vec, weights, base_heuristic, goal_list)
		r_cap = h_cap_parent - h_cap_child
		e_dot = (cost - r_cap)/r_cap
		return e_dot

	@staticmethod
	def getHcap(node, feature_vec, weights, base_heuristic, goal_list):
		#From Equation: h_cap = h_base*(e_cap_dot + 1)
		h_base = 0
		for goal in goal_list: #This will work for multiple goals as well
			h_base += base_heuristic(node, goal)
		e_cap_dot = SearchAgent.getEcapDot(feature_vec, weights)
		h_cap = h_base*(e_cap_dot + 1)
		return h_cap
	
	@staticmethod
	def getEcapDot(feature_vec, weights):
		#From Equation: e_cap_dot = w*features
		return np.dot(feature_vec, weights)

	@staticmethod	
	def getCPlusH(cost, next, feature_vec, weights, base_heuristic, goal_list):
		#From Equation: h(s) <= c(s,s') + h(s')
		return cost + SearchAgent.getHcap(next, feature_vec, weights, base_heuristic, goal_list)

	def display(self):
		cv2.namedWindow('Planning', cv2.WINDOW_NORMAL)
		while True:
			cv2.imshow('Planning', self.img)
			cv2.waitKey(30)






