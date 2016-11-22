#!/usr/bin/env python

import numpy as np

def Euclidean(cell, goal):
    return np.linalg.norm(np.array(cell) - np.array(goal))


def Manhattan(cell, goal):
    return np.sum(np.abs(np.array(cell) - np.array(goal)))


def Octile(cell, goal):
    temp = np.abs(np.array(cell) - np.array(goal))
    return max(temp) + 0.414*min(temp)
