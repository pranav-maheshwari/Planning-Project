#!/usr/bin/env python
import os
import sys
import cPickle

sys.path.insert(0, os.path.abspath('..'))

from utils.io import *
from feature_extract import Feature
from planners.Dijkstra import *


def getEnvironmentDatabase(connectivity="four_connected", obstacles="soft", obstacle_cost=10, num_env_to_load=1, preloaded=True, dijkstra=True):
    planning_problems = []
    if preloaded:
        try:
            for i in xrange(num_env_to_load):
                file_handler = open("../../heuristic_learning/environment_database/puddle/" + str(i) + ".p", 'rb')
                g = cPickle.load(file_handler)
                start_list = cPickle.load(file_handler)
                goals_list = cPickle.load(file_handler)
                feature_map = cPickle.load(file_handler)
                dijkstra_feature = cPickle.load(file_handler)
                if dijkstra:
                    if dijkstra_feature:
                        planning_problems.append((g, start_list, goals_list, feature_map, dijkstra_feature))
                    else:
                        raise IOError
                else:
                    planning_problems.append((g, start_list, goals_list, feature_map))
                file_handler.close()
            return planning_problems
        except IOError:
                planning_problems = []
                pass
    for i in xrange(num_env_to_load):
        file_name = "../../heuristic_learning/environment_database/puddle/" + str(i) + ".txt"
        start_list, goals_list, width, height, walls, count, features = read_env_from_file(file_name)
        g = env_to_graph(start_list, goals_list, width, height, walls, connectivity, obstacles, obstacle_cost)
        feature_obj = Feature(features, height, width, count, connectivity)
        feature_map = feature_obj.get_feature()
        dijkstra_feature = []
        if dijkstra:
            parent, dijkstra_feature = dijkstra_search_fill(g, goals_list[0])
            planning_problems.append((g, start_list, goals_list, feature_map, dijkstra_feature))
        else:
            planning_problems.append((g, start_list, goals_list, feature_map))
        file_handler = open("../../heuristic_learning/environment_database/puddle/" + str(i) + ".p", 'wb')
        cPickle.dump(g, file_handler)
        cPickle.dump(start_list, file_handler)
        cPickle.dump(goals_list, file_handler)
        cPickle.dump(feature_map, file_handler)
        cPickle.dump(dijkstra_feature, file_handler)
        file_handler.close()
    return planning_problems
