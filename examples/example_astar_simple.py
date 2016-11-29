#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


from utils.io import *
from graphs.GridWithWeights import GridWithWeights
from planners.Dijkstra import *
from utils.planner_utils import *
from planners.AStar import *

# Read environment definition from file
start_list, goal_list, width, height, walls = read_env_from_file("../sample_environments/1trap.txt")
print start_list, goal_list, width, height
# Initialize a graph with four or eight connectivity
g = GridWithWeights(width, height, "four_connected")
g.walls = walls
# Set the weights using for loop
for i in xrange(width):
    for j in xrange(height):
        if (i, j) in g.walls:
            continue
        g.weights[(i, j)] = 1  # Uniform cost graph

weights = np.array([0, 20, 0, 400])

# w[0] = g-value
# w[1] = distance-to-goal estimate
# w[2] = depth
# w[3] = nearest obstacle

# Run astar planner and get parents
parents, cost_so_far = astar_search(g, start_list[0], goal_list[0], Euclidean, True, weights)
print(parents[goal_list[0]])
path = reconstruct_path(parents, start_list[0], goal_list[0], cost_so_far)
print("output Path")
print(path)
# [TODO: Add code for visualization]
