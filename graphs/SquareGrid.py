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
        print("Really am here")
        (x, y) = id
        if self.connectivity == "four_connected":
            temp = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
            if (x + y) % 2 == 0: temp.reverse() # aesthetics
            results = filter(self.passable, temp)
            obs_neighbors = set(temp) - set(results)
            results = filter(self.in_bounds, list(results))
        elif self.connectivity == "eight_connected":
            temp = [(x+1, y), (x+1, y-1), (x, y-1), (x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]
            if (x + y) % 2 == 0: temp.reverse() # aesthetics
            results = filter(self.passable, temp)
            obs_neighbors = set(temp) - set(results)
            results = filter(self.in_bounds, list(results))
        return results, obs_neighbors
