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


def get_next_node(v, map):
    x1, y1, f = v
    rows, cols = map.shape
    check_next_node = lambda x, y: True if 0 <= y < cols and 0 <= x < rows and not bool(map[x][y]) else False
    lenght = lambda x, y: math.sqrt(math.fabs(x) + math.fabs(y))
    if f == 1:
        ways =  [0, 1, 1], [-1, 1, 8], [1, 1, 5], [0, 0, 8], [0, 0, 5]
    elif f == 2:
        ways =  [1, 1, 5], [1, 0, 2], [1, -1, 6], [0, 0, 5],[0, 0, 6]
    elif f == 3:
        ways =  [1, -1, 6], [0, -1, 3], [-1, -1, 7], [0, 0, 6], [0, 0, 7]
    elif f == 4:
        ways =  [-1, 1, 8], [-1, 0, 4], [-1, -1, 7], [0, 0, 8],[0, 0, 7]
    elif f == 5:
        ways =  [0, 1, 1], [1, 1, 5], [1, 0, 2], [0, 0, 1], [0, 0, 2]
    elif f == 6:
        ways =  [1, 0, 2], [1, -1, 6], [0, -1, 3], [0, 0, 2], [0, 0, 3]
    elif f == 7:
        ways =  [-1, 0, 4], [-1, -1, 7], [0, -1, 3], [0, 0, 4], [0, 0, 3]
    elif f == 8:
        ways =  [-1, 0, 4], [1, 1, 8], [0, 1, 1], [0, 0, 4], [0, 0, 1]

    return [((x1 + dx, y1 + dy, df), round(lenght(dx, dy) * 10)) for dx, dy, df in ways if check_next_node(x1 + dx, y1 + dy)]

def astar_search(map, start, end):
    open = []
    closed = []

    start_node = Node(start, None)
    goal_node = Node(end, None)

    open.append(start_node)

    while len(open) > 0:
        open.sort()
        current_node = open.pop(0)
        closed.append(current_node)

        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.position)
                current_node = current_node.parent
            # path.append(start)
            # Return reversed path
            return path[::-1]

        (x, y, f) = current_node.position
        #neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        neighbors = get_next_node((x,y, f), map)
        # print('current_node',current_node.position)
        # print('neighbors', neighbors)
        # Loop neighbors
        for next, weight in neighbors:
            # Get value from map
            # print(next)

            # Create a neighbor node
            neighbor = Node(next, current_node)
            # Check if the neighbor is in the closed list
            if (neighbor in closed):
                continue

            neighbor.g = neighbor.g + weight / 10
            neighbor.h = ((neighbor.position[0] - goal_node.position[0])**2 + (neighbor.position[1] - goal_node.position[1])**2)
            # neighbor.h = abs(neighbor.position[0] - goal_node.position[0]) + abs(neighbor.position[1] - goal_node.position[1])
            # neighbor.h = max(abs(neighbor.position[0] - goal_node.position[0]), abs(neighbor.position[1] - goal_node.position[1]))*2
            neighbor.f = neighbor.g + neighbor.h
            # Check if neighbor is in open list and if it has a lower f value
            if (add_to_open(open, neighbor) == True):
                open.append(neighbor)
    # Return None, no path is found
    return None


# Check if a neighbor should be added to open list
def add_to_open(open, neighbor):
    for node in open:
        if (neighbor == node and neighbor.f >= node.f):
            return False
    return True

def solve(bool_map, Start, Goal):
    # Map = genfromtxt('my_data.csv', delimiter=',', dtype= int )
    #graph = graf.create_graf_2(Map)
    # Start = tuple(reversed(Start))
    # Goal = tuple(reversed(Goal))
    start = Start
    goal = Goal
    path = astar_search(bool_map, start, goal)
    print(path)
    return path