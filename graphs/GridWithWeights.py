#!/usr/bin/env python
from SquareGrid import SquareGrid

class GridWithWeights(SquareGrid):
    def __init__(self, width, height, connectivity = "four_connected"):
        SquareGrid.__init__(self, width, height, connectivity)
        self.weights = {}
    
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)