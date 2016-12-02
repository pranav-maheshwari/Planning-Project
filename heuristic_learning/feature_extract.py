import Queue
import numpy as np
from collections import defaultdict

class Feature:
    def __init__(self, features, size_y, size_x, count, normalize=True, connectivity="four_connected"):
        self._connectivity = connectivity
        self.four_connected = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.eight_connected = [(-1, 0), (1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        self.limit_x = size_x
        self.limit_y = size_y
        self.feature_lookup = [dict()]*count
        self.feature = defaultdict(lambda: (size_y + size_x))
        n = 1
        if normalize:
            n = 64
        if count > 0:
            for i in range(count):
                self.feature_lookup[i] = self.BFS(self.initiate_open_list(features[i]), self.initiate_grid(features[i]))
            for i in self.feature_lookup[0].iterkeys():
                self.feature[i] = tuple((1.0*d[i])/n for d in self.feature_lookup)

    def initiate_open_list(self, feature):
        open_list = list()
        for i in range(feature[1], feature[3]):
            open_list.append((feature[0], i))
            open_list.append((feature[2], i))
        for i in range(feature[0] + 1, feature[2]):
            open_list.append((i, feature[1]))
            open_list.append((i, feature[3]))
        return open_list

    def initiate_grid(self, feature):
        grid = np.ones((self.limit_y, self.limit_x), dtype=int)
        for i in range(feature[1], feature[3]):
            for j in range(feature[0] + 1, feature[2]):
                grid[j][i] = -1
        return grid

    def BFS(self, open_list, grid):
        expand = Queue.Queue()
        tree = dict()
        for i in open_list:
            expand.put(i)
            tree[i] = 0
        while not expand.empty():
            i = expand.get()
            for j in self.successors(i):
                if j not in tree:
                    tree[j] = tree[i] + grid[j[0]][j[1]]
                    expand.put(j)
        return tree

    def successors(self, node):
        successors = list()
        if self._connectivity == "four_connected":
            for i in self.four_connected:
                temp_y = node[0] + i[0]
                temp_x = node[1] + i[1]
                if 0 <= temp_x < self.limit_x and 0 <= temp_y < self.limit_y:
                    successors.append((temp_y, temp_x))
            return successors
        elif self._connectivity == "eight_connected":
            for i in self.eight_connected:
                temp_y = node[0] + i[0]
                temp_x = node[1] + i[1]
                if 0 <= temp_x < self.limit_x and 0 <= temp_y < self.limit_y:
                    successors.append((temp_y, temp_x))
            return successors

    def get_feature(self):
        return self.feature
