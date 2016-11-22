import cv2
import numpy as np
from collections import namedtuple
import random

_config = namedtuple('_config', 'res_x res_y type count width depth')

_config.res_x = 64          # Resolution in X axis
_config.res_y = 64          # Resolution in Y axis
_config.type = "bars"       # Map type (trap or bars)
_config.count = 2           # Number of bars
_config.width = 32          # Width of trap
_config.depth = 32          # Depth of trap
_config.length = 24         # Length of bars
_config.span = 0.6          # Ratio of region occupied by bars or trap

class Example:
    def __init__(self):
        self.output = np.ones([_config.res_x, _config.res_y], dtype=int) * 255
        if _config.type == "bars":
            for i in range(0, _config.count):
                x = random.randint(0, _config.res_x)
                y = random.randint(int((1-_config.span)*_config.res_y), int(_config.span*_config.res_y))
                self.output = cv2.line(self.output, (x, y - _config.length/2), (x, y + _config.length/2), (0, 0, 0), thickness=4)
        elif _config.type == "trap":
            x = random.randint(int((1-_config.span)*_config.res_x), int(_config.span*_config.res_x))
            y = random.randint(int((1-_config.span)*_config.res_y), int(_config.span*_config.res_y))
            vertex = np.array([[x - _config.depth/2, y - _config.width/2], [x + _config.depth/2, y - _config.width/2], [x + _config.depth/2, y + _config.width/2], [x - _config.depth/2, y + _config.width/2]], dtype=np.int32)
            self.output = cv2.polylines(self.output, pts=[vertex], color=(0, 0, 0), isClosed=False)

    def display(self):
        cv2.imwrite("current.jpg", self.output)

    def access(self):
        return self.output

test = Example()
test.display()
