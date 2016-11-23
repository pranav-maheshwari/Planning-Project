#!/usr/bin/env python
import numpy as np


def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    print came_from
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start) # optional
    path.reverse() # optional
    return path


def GetNearestObstacle(cobs, config, distance="Euclidean"):
    dists = []
    temp = np.array(config)
    for v in cobs:
        if distance == "Euclidean":
            dists.append(np.linalg.norm(temp - np.array(v)))
        else:
            dists.append(np.sum(np.abs(np.array(temp) - np.array(v))))
    return min(dists)
