#!/usr/bin/env python
import os
import sys
import cv2
import cv
import matplotlib.pylab as plt
from threading import Thread

sys.path.insert(0, os.path.abspath('..'))

from data_structures.PriorityQueue import *
from graphs.GridWithWeights import *
from graphs.HeuristicFunctions import *

img = np.array([0])


def display():
    cv2.namedWindow('AutoPark', cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow("Planning", img)
        cv2.waitKey(30)

t1 = Thread(target=display)


def astar_search(graph, start, goal, heuristic, visualize, weight):
    global img
    if visualize:
        img = np.ones([graph.width, graph.height])*255
        for i in graph.walls:
            img[i[0], i[1]] = 0
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        t1.start()
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    while not frontier.empty():
        current = frontier.get()
        img[current[0], current[1]] = [255, 0, 0]
        # print(current)
        if current == goal:
            break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def astar_search_multistart(graph, start_list, goal, heuristic, weight):
    """AStar search with multiple start points but single goal point"""
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    for start in start_list:
        frontier.put(start, 0)
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
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far
