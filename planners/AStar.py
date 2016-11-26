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

img = np.array([0])


def display():
    cv2.namedWindow('Planning', cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow('Planning', img)
        cv2.waitKey(30)

t1 = Thread(target=display)


def astar_search(graph, start, goal, heuristic, visualize, weights):
    global img
    if visualize:
        img = np.ones([graph.width, graph.height, 3])*255
        for i in graph.walls:
            img[i[0], i[1]] = (0, 0, 0)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t1.start()
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    depth_so_far = {}
    obs_so_far = set()
    came_from[start] = None
    cost_so_far[start] = 0
    depth_so_far[start] = 0
    while not frontier.empty():
        current = frontier.get()
        img[current[0], current[1]] = [255, 0, 0]
        # print(current)
        if current == goal:
            break
        neighbors, obs_neighbors = graph.neighbors(current)
        #Add obstacle neighbors to obs_so_far
        for obs_neighbor in obs_neighbors:
            obs_so_far.add(obs_neighbor) 
        for next in neighbors:
            #Add any obstacles seen to obs_so_far
            new_cost = cost_so_far[current] + graph.cost(current, next)
            new_depth = depth_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                depth_so_far[next] = new_depth
                feature_array = getNodeFeatures(next, goal, heuristic, new_cost, obs_so_far, new_depth)
                priority = new_cost + np.dot(weights, feature_array)
                frontier.put(next, priority)
                came_from[next] = current
        time.sleep(0.1)
    return came_from, cost_so_far


def astar_search_multistart(graph, start_list, goal, heuristic, weight):
    """AStar search with multiple start points but single goal point"""
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    for start in start_list:
        frontier.put((start, 0, 0), 0)
        came_from[start] = None
        cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put((next, new_cost, current[0][2] + 1), priority)
                came_from[next] = current

    return came_from, cost_so_far
