#!/usr/bin/env python

import cv2
import numpy as np
from collections import namedtuple
import random
import os

_config = namedtuple('_config', 'res_x res_y type count width depth')

_config.res_x = 64          # Resolution in X axis
_config.res_y = 64          # Resolution in Y axis
_config.type = "puddle"       # Map type (trap or bars or puddle)
_config.count = 2           # Number of bars
_config.length = 32         # Length of bars
_config.start = [40, 25]    # Start Position
_config.goal = [40, 50]     # Goal Position
_config.thickness = 5       # Thickness of walls
_config.depth = abs(_config.goal[1] - _config.start[1] - _config.thickness - 2)             # Max Width of trap
_config.width = 32          # Max Depth of trap
_config.puddle_size = 10    # Max puddle size
_config.puddle_count = 10   # Max number of puddles
_config.boundaries_x = sorted([_config.start[1], _config.goal[1]])                      # Region occupied by bars or trap
_config.boundaries_y = sorted([_config.start[0], _config.goal[0]])                      # Region occupied by bars or trap

class Example:
    def __init__(self, count, param, disp):
        for i in range(count):
            f = open(os.path.join(_config.type,str(i)) + ".txt", 'w')
            buffer_temp = param[0:2]
            if disp:
                self.output = np.ones([_config.res_x, _config.res_y], dtype=int) * 255
                self.output[_config.start[0]][_config.start[1]] = 0
                self.output[_config.goal[0]][_config.goal[1]] = 0
            if _config.type == "bars":
                walls = list()
                for k in range(0, _config.count):
                    length = max(3*_config.thickness, int(random.random()*_config.length))
                    x = random.randint(_config.boundaries_x[0], _config.boundaries_x[1])
                    y = random.randint(0, _config.res_y - length)
                    walls.append([x, y, x + _config.thickness, y + length])
                if disp:
                    for k in walls:
                        cv2.rectangle(self.output, tuple(k[0:2]), tuple(k[2:]), (0, 0, 0), -1)
            elif _config.type == "bugtrap_environments":
                print i
                depth = max(2*_config.thickness, int(random.random()*_config.depth))
                width = max(2*_config.thickness, int(random.random()*_config.width))
                print "Depth", depth
                print "Limits", _config.boundaries_x[0]+1, _config.boundaries_x[1] - depth - _config.thickness - 1
                top_left_x = random.randint(_config.boundaries_x[0]+1, _config.boundaries_x[1] - depth - _config.thickness - 1)
                print top_left_x
                top_left_y = random.randint(0, _config.res_y - 2*_config.thickness - width)
                walls = [[top_left_x, top_left_y, top_left_x + depth + _config.thickness, top_left_y + _config.thickness], [top_left_x + depth, top_left_y + _config.thickness, top_left_x + depth + _config.thickness, top_left_y + width + _config.thickness], [top_left_x, top_left_y + width + _config.thickness, top_left_x + depth + _config.thickness, top_left_y + width + 2*_config.thickness], [top_left_x, top_left_y + _config.thickness, top_left_x + _config.thickness, top_left_y + width + _config.thickness]]
                walls.pop(random.randint(0, 3))
                if disp:
                    for k in walls:
                        cv2.rectangle(self.output, tuple(k[0:2]), tuple(k[2:]), (0, 0, 0), -1)
            elif _config.type == "puddle":
                walls = list()
                puddle_count = int(random.random() * _config.puddle_count)
                for k in range(puddle_count):
                    puddle_size = int(max(0.1, random.random())*_config.puddle_size)
                    x = random.randint(0, _config.res_x - puddle_size - 1)
                    y = random.randint(0, _config.res_y - puddle_size - 1)
                    while not self.clearance_check(x, y, puddle_size):
                        x = random.randint(0, _config.res_x)
                        y = random.randint(0, _config.res_y)
                    walls.append([x, y, x + puddle_size, y + puddle_size])
                    if disp:
                        for j in walls:
                            cv2.rectangle(self.output, tuple(j[0:2]), tuple(j[2:]), (0, 0, 0), -1)
            for k in walls:
                k = [k[1], k[0], k[3], k[2]]
                buffer_temp.append(str(k)[1:-1])
            if _config.type == "puddle":
                buffer_temp.append("count " + str(puddle_count))
            buffer_temp += param[-2:]
            for k in buffer_temp:
                f.write(k)
                f.write("\n")
            f.close()
            if disp:
                self.display(i)

    def display(self, count):
        cv2.imwrite(os.path.join(_config.type, str(count)) + ".jpg", self.output)

    def clearance_check(self, x, y, size):
        if x <= _config.start[1] <= x + size or y <= _config.start[0] <= y + size or x <= _config.goal[1] <= x + size or y <= _config.goal[0] <= y + size:
            return False
        else:
            return True

def static_strings():
    output = list()
    output.append("width " + str(_config.res_x))
    output.append("height " + str(_config.res_y))
    output.append("start " + str(_config.start)[1:-1])
    output.append("goal " + str(_config.goal)[1:-1])
    return output

parameters = static_strings()
test = Example(100, parameters, True)
