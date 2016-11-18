#!/usr/bin/env python

"""a reasonably fast priority queue that uses binary heaps, 
but does not support reprioritize. To get the right ordering, 
we use tuples (priority, item). When an element is inserted that is already in the queue, 
we have a duplicate"""
import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]