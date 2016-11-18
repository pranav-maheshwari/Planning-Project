#!/usr/bin/env python


def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    print came_from
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start) # optional
    path.reverse() # optional
    return path