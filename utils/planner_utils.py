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


def normalize(feature, MAX, MIN):
    try:
        temp = np.divide(feature - MIN, MAX - MIN)
    except RuntimeWarning:
        temp = feature
    return temp


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
        f_obs = getNearestObstacle(obs_so_far, next)
    return np.array([f_cost, f_h, f_depth, f_obs])


def getEdgeFeatures(parent, child, goal_list, dist_to_goal_fn, cost_so_far, c_obs, tree_depths):
    f_cost = cost_so_far[parent]
    f_h = 0
    for goal in goal_list: #This will work for multiple goals as well
        f_h += dist_to_goal_fn(parent, goal)
    f_depth = tree_depths[parent]
    # f_length = dist_to_goal_fn(parent, child)
    if len(c_obs) == 0:
        f_obs = 0
    else:
        # f_obs = getNearestObstacle(c_obs, parent)
        f_obs = getNearestObstacle(c_obs, parent)
    feature_vec = np.array([f_obs, f_h])
    # feature_vec = np.array([f_cost, f_h, f_depth, f_obs])
    # print(f_cost, f_h, f_depth, f_obs)
    # print(feature_vec)
    return feature_vec#, f_length])


