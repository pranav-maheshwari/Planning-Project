#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from data_structures.PriorityQueue import *
from graphs.GridWithWeights import *


def astar_search(graph, start, goal, heuristic):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    while not frontier.empty():
        current = frontier.get()
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


def astar_search_multistart(graph, start_list, goal, heuristic):
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
