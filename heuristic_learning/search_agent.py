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
	def __init__(self, graph, start_list, goal_list, base_heuristic, w_curr, visualize=True):
		self.graph = graph
		self.start_list = start_list
		self.goal_list = goal_list
		self.base_heuristic = base_heuristic
		self.w_curr = w_curr
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
	
	def setWeights(self, w):
		self.w_curr = w

	def step(self, action):
		current = self.frontier.get()

	def display(self):
		cv2.namedWindow('Planning', cv2.WINDOW_NORMAL)
		while True:
			cv2.imshow('Planning', self.img)
			cv2.waitKey(30)






