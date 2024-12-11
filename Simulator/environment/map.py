import pygame as pg
import numpy as np
import sys
import os

import copy

import cv2 


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from environment.obstacles import *


class Map:
    def __init__(self, size = (200, 200)) -> None:
        self.obs_param = 200
        self.size = size
        self.map = np.zeros(self.size)
        self.map_d = np.zeros(self.size)
        self.bool_map = np.zeros(self.size)
        self.scale = 0.1
        self.expansion_cells = 3
        self.resized_map = cv2.resize(self.bool_map, (int(self.size[0]*self.scale) , int(self.size[1]*self.scale)), interpolation = cv2.INTER_NEAREST)
        self.obstacles = []
        pass

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def get_obstacles(self):
        return self.obstacles

    def generate(self):
        def get_bool_map(map):
            bool_map = [ [False for j in range(len(map[0]))] for i in range(len(map))]
            for i in range(len(map)):
                for j in range(len(map[0])):
                    if map[i][j] >= 0.5:
                        bool_map[i][j] = 1
                    else:
                        bool_map[i][j] = 0
            return bool_map
        
        
        def extend_bool_map(bool_map):
            def get_cels_in_radius(v, map):
                x1, y1 = v
                rows, cols = map.shape
                check_next_node = lambda x, y: True if 0 <= y < cols and 0 <= x < rows and not bool(map[x][y]) else False
                ways = [-1, 0], [0, -1], [1, 0], [0, 1], [-1, -1], [1, -1], [1, 1], [-1, 1]
                return [(x1 + dx, y1 + dy) for dx, dy in ways if check_next_node(x1 + dx, y1 + dy)]
                
            extended_bool_map = copy.copy(bool_map)


            return extended_bool_map
   

    def update(self):
        # self.path = solve(self.resized_map, (0 , 0), (10 , 10))
        pass

    def draw(self, screen):
        for obstacle in self.obstacles:
            obstacle.draw(screen)
        # surf = pg.surfarray.make_surface(self.map)
        # screen.blit(surf, (0, 0))


