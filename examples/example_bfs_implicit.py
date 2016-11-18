#! /usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from utils.io import *
from graphs.SquareGrid import SquareGrid
from planners.bfs import *

#Read environment definition from file
start_list, goal_list, width, height, walls = read_env_from_file("../sample_environments/sample_simple.txt")
print start_list, goal_list, width, height
#Initialize a graph with four or eight connectivity
g = SquareGrid(width, height, "four_connected")
g.walls = walls
#Run bfs planner and get parents
parents = breadth_first_search_implicit_with_goal(g, start_list[0], goal_list[0])
print("Done Search")
#Print output
# print parents
#[TODO: Add code for visualization]