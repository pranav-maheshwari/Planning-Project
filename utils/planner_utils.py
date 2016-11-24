#!/usr/bin/env python
import numpy as np


def reconstruct_path(came_from, start, goal, cost_so_far):
    current = goal
    path = [current]
    path_cost = cost_so_far[current]
    while current != start:
        current = came_from[current]
        path_cost += cost_so_far[current]
        path.append(current)
    path.append(start) # optional
    path.reverse() # optional
    return path, path_cost


def getNearestObstacle(cobs, config, distance="euclidean"):
    dists = []
    temp = np.array(config)
    if distance == "euclidean":
        dists = [np.linalg.norm(temp - np.array(v)) for v in cobs]
    else:
        dists = [np.sum(np.abs(np.array(temp) - np.array(v))) for v in cobs]
    return min(dists)

def getNodeFeatures(next, goal, heuristic, new_cost, obs_so_far, new_depth):
    f_cost = new_cost
    f_h = heuristic(next, goal)
    f_depth = new_depth
    if len(obs_so_far) == 0:
        f_obs = 0
    else:
        f_obs = 1.0/(0.0001 + getNearestObstacle(obs_so_far, next))
    return np.array([f_cost, f_h, f_depth, f_obs])

