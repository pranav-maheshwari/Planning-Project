#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from graphs.SimpleGraph import SimpleGraph
from data_structures.Queue import Queue


def breadth_first_search_explicit(graph, start, goal):
    """Doe breadth first search using an explicit graph""" 
    # print out what we find
    frontier = Queue()
    frontier.put(start)
    visited = {}
    visited[start] = True
    
    while not frontier.empty():
        current = frontier.get()
        print("Visiting %r" % current)
        for next in graph.neighbors(current):
            if next not in visited:
                frontier.put(next)
                visited[next] = True

def breadth_first_search_implicit(graph, start):
    # return "came_from"
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None
    while not frontier.empty():
        # print "Searching ..."
        current = frontier.get()
        for next in graph.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current
    
    return came_from

def breadth_first_search_implicit_with_goal(graph, start, goal):
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None
    while not frontier.empty():
        # print "Searching ..."
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current
    
    return came_from

