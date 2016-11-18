#! /usr/bin/env python
"""Constructs a simple explicit graph and calls bfs planner on it"""
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from graphs.SimpleGraph import SimpleGraph
from planners.bfs import *

example_graph = SimpleGraph()
example_graph.edges = {
    'A': ['B'],
    'B': ['A', 'C', 'D'],
    'C': ['A'],
    'D': ['E', 'A'],
    'E': ['B']
}

breadth_first_search_explicit(example_graph, 'A')

