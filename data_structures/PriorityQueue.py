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
    
    def put(self, item, priority, tie_break, tie_break_2):
        heapq.heappush(self.elements, (priority, tie_break, tie_break_2, item))
    
    def get(self):
    	e = heapq.heappop(self.elements)
        return e[3], e[0]