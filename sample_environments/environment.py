import cv2
import numpy as np
from collections import namedtuple
import random

_config = namedtuple('_config', 'res_x res_y type count width depth')

_config.res_x = 64          # Resolution in X axis
_config.res_y = 64          # Resolution in Y axis
_config.type = "trap"       # Map type (trap or bars)
_config.count = 2           # Number of bars
_config.width = 32          # Max Width of trap
_config.depth = 32          # Max Depth of trap
_config.length = 24         # Length of bars
_config.start = [40, 25]    # Start Position
_config.goal = [40, 50]     # Goal Position
_config.boundaries_x = sorted([_config.start[1], _config.goal[1]])   # Region occupied by bars or trap
_config.boundaries_y = sorted([_config.start[0], _config.goal[0]])   # Region occupied by bars or trap
_config.thickness = 5       # Thickness of walls

class Example:
    def __init__(self, count, param, disp):
        for i in range(count):
            f = open(str(i)+_config.type+".txt", 'w')
            buffer_temp = param[0:2]
            if disp:
                self.output = np.ones([_config.res_x, _config.res_y], dtype=int) * 255
            if _config.type == "bars":
                """
                for i in range(0, _config.count):
                    x = random.randint(0, _config.res_x)
                    y = random.randint(int((1-_config.span)*_config.res_y), int(_config.span*_config.res_y))
                    if disp:
                        self.output = cv2.line(self.output, (x, y - _config.length/2), (x, y + _config.length/2), (0, 0, 0), thickness=4)
                """
            elif _config.type == "trap":
                depth = max(2*_config.thickness, int(random.random()*_config.depth))
                width = max(2*_config.thickness, int(random.random()*_config.width))
                top_left_x = random.randint(_config.boundaries_x[0], min(_config.boundaries_x[0] + depth, _config.boundaries_x[1] - 2*_config.thickness))
                top_left_y = random.randint(0, _config.res_y - 2*_config.thickness - width)
                walls = [[top_left_y, top_left_x, top_left_y + _config.thickness, top_left_x + depth + _config.thickness], [top_left_y + _config.thickness, top_left_x + depth, top_left_y + width + _config.thickness, top_left_x + depth + _config.thickness], [top_left_y + width + _config.thickness, top_left_x, top_left_y + width + 2*_config.thickness, top_left_x + depth + _config.thickness], [top_left_y + _config.thickness, top_left_x, top_left_y + width + _config.thickness, top_left_x + _config.thickness]]
                walls.pop(random.randint(0, 3))
                if disp:
                    for k in walls:
                        cv2.rectangle(self.output, tuple(k[0:2]), tuple(k[2:]), (0, 0, 0), -1)
            for k in walls:
                buffer_temp.append(str(k)[1:-1])
            buffer_temp += param[-2:]
            for k in buffer_temp:
                f.write(k)
                f.write("\n")
            f.close()
            if disp:
                self.display(i)

    def display(self, count):
        cv2.imwrite(str(count)+_config.type + ".jpg", self.output)

def static_strings():
    output = list()
    output.append("width " + str(_config.res_x))
    output.append("height " + str(_config.res_y))
    output.append("start " + str(_config.start)[1:-1])
    output.append("goal " + str(_config.goal)[1:-1])
    return output

parameters = static_strings()
test = Example(100, parameters, True)
