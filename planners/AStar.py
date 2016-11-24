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
    frontier.put((start, 0, 0), 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    obs_list = set()
    while not frontier.empty():
        current = frontier.get()
        img[current[0][0], current[0][1]] = [255, 0, 0]
        # print(current)
        time.sleep(0.1)
        if current[0] == goal:
            break
        successors, temp = graph.neighbors(current[0])
        for i in temp:
            obs_list.add(i)
        for next in successors:
            new_cost = cost_so_far[current[0]] + graph.cost(current[0], next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                came_from[next] = current[0]
                next = (next, new_cost, current[2] + 1)
                if len(obs_list) == 0:
                    fobs = 0
                else:
                    fobs = weights[3]*(1.0/(0.0001 + GetNearestObstacle(obs_list, next[0])))
                print fobs, weights[0]*heuristic(next[0], goal)
                priority = new_cost + weights[0]*heuristic(next[0], goal) + fobs + Feature(next, weights)
                print priority
                frontier.put(next, priority)
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
