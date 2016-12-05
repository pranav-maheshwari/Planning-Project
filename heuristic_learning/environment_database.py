#!/usr/bin/env python
import os
import sys
import cPickle

sys.path.insert(0, os.path.abspath('..'))

from utils.io import *
import feature_extract 
from planners.Dijkstra import *
from graphs.HeuristicFunctions import *


def getEnvironmentDatabase(connectivity="four_connected", obstacles="soft", obstacle_cost=10, num_env_to_load=1, preloaded=True, dijkstra=True, need_additional = True, need_normalized = True):
    planning_problems = []
    if preloaded:
        try:
            const = ""
            if need_additional: 
                const = "NA"
            for i in xrange(num_env_to_load):
                file_handler = open("../../heuristic_learning/environment_database/bugtrap_environments/" + const + str(i) + ".p", 'rb')
                print "../../heuristic_learning/environment_database/bugtrap_environments/" + const + str(i) + ".p"
                print "Loading ", i
                g = cPickle.load(file_handler)
                start_list = cPickle.load(file_handler)
                goals_list = cPickle.load(file_handler)
                feature_map = cPickle.load(file_handler)
                dijkstra_feature = cPickle.load(file_handler)
                if dijkstra:
                    if dijkstra_feature:
                        planning_problems.append((g, start_list, goals_list, feature_map, dijkstra_feature))
                    else:
                        raise EOFError
                else:
                    planning_problems.append((g, start_list, goals_list, feature_map))
                file_handler.close()
            return planning_problems
        except (EOFError, IOError):
                planning_problems = []
                print "Reverting to regular computing of environments"
                pass
    for i in xrange(num_env_to_load):
        const = ""
        if need_additional: 
            const = "NA"
        file_name = "../../heuristic_learning/environment_database/bugtrap_environments/" + str(i) + ".txt"
        start_list, goals_list, width, height, walls, count, features = read_env_from_file(file_name)
        feature_extract.size_y = height
        feature_extract.size_x = width
        g = env_to_graph(start_list, goals_list, width, height, walls, connectivity, obstacles, obstacle_cost)
        feature_obj = feature_extract.Feature(features, goals_list[0], height, width, count, need_normalized, need_additional)
        feature_map = feature_obj.get_feature()
        dijkstra_feature = []
        if dijkstra:
            parent, dijkstra_feature = dijkstra_search_fill(g, goals_list[0])
            planning_problems.append((g, start_list, goals_list, feature_map, dijkstra_feature))
        else:
            planning_problems.append((g, start_list, goals_list, feature_map))
        file_handler = open("../../heuristic_learning/environment_database/bugtrap_environments/" + const + str(i) + ".p", 'wb')
        cPickle.dump(g, file_handler)
        cPickle.dump(start_list, file_handler)
        cPickle.dump(goals_list, file_handler)
        cPickle.dump(feature_map, file_handler)
        cPickle.dump(dijkstra_feature, file_handler)
        file_handler.close()
    return planning_problems
