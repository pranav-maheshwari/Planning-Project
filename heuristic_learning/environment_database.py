#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
from utils.io import *

NUM_BUGTRAP = 50
def getEnvironmentDatabase():
	graphs = []
	for i in xrange(NUM_BUGTRAP):
		file_name = "../heuristic_learning/environment_database/bugtrap_environments/" + str(i) + ".txt"
		start_list, goals_list, width, height, walls = read_env_from_file(file_name)
		g = env_to_graph(start_list, goals_list, width, height, walls)
		graphs.append((g,start_list, goals_list))
	return graphs

