#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from utils.io import *
from feature_extract import Feature




def getEnvironmentDatabase(connectivity = "four_connected", obstacles = "soft", obstacle_cost = 10, num_env_to_load = 1):
    planning_problems = []
    for i in xrange(num_env_to_load):
        file_name = "../../heuristic_learning/environment_database/puddle/" + str(i) + ".txt"
        start_list, goals_list, width, height, walls, count, features = read_env_from_file(file_name)
        g = env_to_graph(start_list, goals_list, width, height, walls, connectivity, obstacles, obstacle_cost)
        feature_obj = Feature(features, height, width, count, connectivity)
        feature_map = feature_obj.get_feature()
        planning_problems.append((g, start_list, goals_list, feature_map))  # ,feature_map
    return planning_problems
