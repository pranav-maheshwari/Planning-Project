#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from graphs.GridWithWeights import *
import re


def read_env_from_file(file_name):
    walls = []
    start_list = []
    goals_list = []
    features = []
    width = 0
    height = 0
    count = 0
    f = open(file_name, 'r')
    for line in f:
        coords = re.split('\W+', line)
        if coords[-1] == '':
            coords = coords[0:-1]
        if coords[0] == 'width':
            width = int(coords[1])
            continue
        elif coords[0] == 'height':
            height = int(coords[1])
            continue
        elif coords[0] == 'start':
            for i in xrange(1, len(coords), 2):
                start_list.append((int(coords[i]), int(coords[i + 1])))
            continue
        elif coords[0] == 'goal':
            for i in xrange(1, len(coords), 2):
                goals_list.append((int(coords[i]), int(coords[i + 1])))
            continue
        elif coords[0] == 'count':
            count = int(coords[1])
        else:
            features.append([int(i) for i in coords])
            for i in xrange(int(coords[0]), int(coords[2]) + 1):
                for j in xrange(int(coords[1]), int(coords[3]) + 1):
                    walls.append((i, j))
    return start_list, goals_list, width, height, walls, count, features


def env_to_graph(start_list, goals_list, width, height, walls):
    g = GridWithWeights(width, height, "four_connected")
    g.walls = walls
    # Set the weights using for loop
    for i in xrange(width):
        for j in xrange(height):
            if (i, j) in g.walls:
                g.weights[(i,j)] = 10
            else:
                g.weights[(i, j)] = 1  # Uniform cost graph
    return g
