#!/usr/bin/env python

class GridWithWeightsSoftObs():
    def __init__(self, width, height, connectivity = "four_connected"):
        self.width = width
        self.height = height
        self.connectivity = connectivity
        self.walls = []
        self.weights = {}

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

    def neighbors(self, id):
		(x, y) = id
		if self.connectivity == "four_connected":
		    temp = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
		    if (x + y) % 2 == 0: temp.reverse() # aesthetics
		    results = filter(self.in_bounds, temp)
		elif self.connectivity == "eight_connected":
		    temp = [(x+1, y), (x+1, y-1), (x, y-1), (x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]
		    if (x + y) % 2 == 0: temp.reverse() # aesthetics
		    results = filter(self.in_bounds, temp)
		return results

