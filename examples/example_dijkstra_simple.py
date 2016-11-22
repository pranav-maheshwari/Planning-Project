#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from utils.io import *
from graphs.GridWithWeights import GridWithWeights
from planners.Dijkstra import *
from utils.planner_utils import *

# Read environment definition from file
start_list, goal_list, width, height, walls = read_env_from_file("../sample_environments/sample_simple.txt")
print start_list, goal_list, width, height
# Initialize a graph with four or eight connectivity
g = GridWithWeights(width, height, "four_connected")
g.walls = walls
# Set the weights using for loop
for i in xrange(width):
    for j in xrange(height):
        if (i, j) in g.walls:
            continue
        g.weights[(i, j)] = 1;  # Uniform cost graph

# Run bfs planner and get parents
parents, cost_so_far = dijkstra_search(g, start_list[0], goal_list[0])
print(parents[goal_list[0]])
path = reconstruct_path(parents, start_list[0], goal_list[0])
print("output Path")
print(path)
# Print output
# print parents
# [TODO: Add code for visualization]
