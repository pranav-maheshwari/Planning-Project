#!/usr/bin/env python
class SquareGrid:
    def __init__(self, width, height, connectivity="four_connected"):
        self.width = width
        self.height = height
        self.connectivity = connectivity
        self.walls = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls
    
    def neighbors(self, id):
        (x, y) = id
        if self.connectivity == "four_connected":
            results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
            if (x + y) % 2 == 0: results.reverse() # aesthetics
            results = filter(self.in_bounds, results)
            results = filter(self.passable, results)
        elif self.connectivity == "eight_connected":
            results = [(x+1, y), (x+1, y-1), (x, y-1), (x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]
            if (x + y) % 2 == 0: results.reverse() # aesthetics
            results = filter(self.in_bounds, results)
            results = filter(self.passable, results)
        return results
