#!/usr/bin/env python
import os
import sys
import cv2
from threading import Thread
import time
from heapq import heapify, heappop, heappush
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


def MHAStar_search(graph, start, goal, heuristic, feature_map, visualize):
    global img
    if visualize:
        img = np.ones([graph.width, graph.height, 3])*255
        for i in graph.walls:
            img[i[0], i[1]] = (0, 0, 0)
        t1.start()
    frontier1 = heapify([])
    frontier2 = heapify([])
    frontier1.put(0, start)
    frontier2.put(0, start)
    frontiers = [frontier1, frontier2]
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0
    closed = set()
    i = 0
    while not frontier1.empty() and frontier2.empty():
        i += 1
        index = i % 2
        priority, current = frontiers[index].heappop()
        img[current[0], current[1]] = [255, 0, 0]
        if current in closed:
            continue
        else:
            closed.add(current)
        if current == goal:
            break
        neighbors = graph.neighbors(current)
        #Add obstacle neighbors to obs_so_far
        for next in neighbors:
            #Add any obstacles seen to obs_so_far
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                for i in range(0, len(frontiers)):
                    feature = feature_map[next].sort()[0:3]
                    priority = heuristic[i].predict(feature)
                    frontiers[i].put(next, priority)
                came_from[next] = current
        time.sleep(0.1)
    return came_from, cost_so_far
