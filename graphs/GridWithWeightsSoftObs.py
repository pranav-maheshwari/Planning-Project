#!/usr/bin/env python
from SquareGrid import SquareGrid

class GridWithWeightsSoftObs(SquareGrid):
    def __init__(self, width, height, connectivity = "four_connected"):
        SquareGrid.__init__(self, width, height, connectivity)
        self.weights = {}
    
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

	def neighbors(self, id):
		(x, y) = id
		if self.connectivity == "four_connected":
		    temp = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
		    if (x + y) % 2 == 0: temp.reverse() # aesthetics
		    results = filter(self.in_bounds, list(results))
		elif self.connectivity == "eight_connected":
		    temp = [(x+1, y), (x+1, y-1), (x, y-1), (x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]
		    if (x + y) % 2 == 0: temp.reverse() # aesthetics
		    results = filter(self.in_bounds, list(results))
		return results
