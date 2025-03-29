import numpy as np
from numpy import genfromtxt
import math
# import graf
# This class represents a node
class Node:
    # Initialize the class
    def __init__(self, position: (), parent: ()):
        self.position = position
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to goal node
        self.f = 0  # Total cost

    # Compare nodes
    def __eq__(self, other):
        return self.position == other.position

    # Sort nodes
    def __lt__(self, other):
        return self.f < other.f

    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.position, self.f))


class AStar:
    def __init__(self, map):
        self.map = map

    def find_path(self, start, goal):
        return self.astar_search(start, goal)

    def get_next_node(self, v):
        def lenght(x, y):
            # if  x == 0 or y == 0:
            #     return 1
            # else:
            #     return 1.4
            lenght = math.sqrt(math.fabs(x)**2 + math.fabs(y)**2)
            return lenght
        x1, y1 = v
        rows, cols = self.map.shape
        check_next_node = lambda x, y: True if 0 <= y < cols and 0 <= x < rows and not bool(self.map[x][y]) else False
        ways = [1, 1], [1, -1], [-1, -1], [-1, 1], [0, 1], [1, 0], [0, -1], [-1, 0]
        return [((x1 + dx, y1 + dy), lenght(dx, dy)) for dx, dy in ways if check_next_node(x1 + dx, y1 + dy)]

    def astar_search(self, start, end):
        open = []
        closed = []
        start_node = Node(start, None)
        goal_node = Node(end, None)

        open.append(start_node)

        while len(open) > 0:
            open.sort()
            # print('open sorted = ', open)
            current_node = open.pop(0)
            closed.append(current_node)

            if current_node == goal_node:
                path = []
                while current_node != start_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

            (x, y) = current_node.position
            neighbors = self.get_next_node((x, y))
            # print('current_node', current_node.position)

            for next, weight in neighbors:
                neighbor = Node(next, current_node)
                if neighbor in closed:
                    continue

                neighbor.g = neighbor.g + weight
                # neighbor.h = ((neighbor.position[0] - goal_node.position[0])**2 + (neighbor.position[1] - goal_node.position[1])**2)
                neighbor.h = math.sqrt(math.fabs(neighbor.position[0] - goal_node.position[0])**2 + math.fabs(neighbor.position[1] - goal_node.position[1])**2)
                # neighbor.h = max(abs(neighbor.position[0] - goal_node.position[0]), abs(neighbor.position[1] - goal_node.position[1]))
                neighbor.f = neighbor.g + neighbor.h

                if self.add_to_open(open, neighbor):
                    open.append(neighbor)
        return None

    def add_to_open(self, open, neighbor):
        for node in open:
            if neighbor == node and neighbor.f >= node.f:
                return False
        return True
