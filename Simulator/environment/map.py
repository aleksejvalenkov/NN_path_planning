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
        self.scale = 0.1
        self.expansion_cells = 3
        self.obstacles = []
        # Cтены 
        # Нужен генератор карт
        obstacle_l = Obstacle(init_pos=[30//2, 1000//2, 0.001], init_size=[30, 1000])
        obstacle_r = Obstacle(init_pos=[1800-30//2, 1000//2, 0.001], init_size=[30, 1000])
        obstacle_t = Obstacle(init_pos=[1800//2, 30//2, 0.001], init_size=[1800, 30])
        obstacle_b = Obstacle(init_pos=[1800//2, 1000-30//2, 0.001], init_size=[1800, 30])

        obstacle_0 = Obstacle(init_pos=[200, 330//2+30, 0.001], init_size=[30, 330])
        obstacle_1 = Obstacle(init_pos=[200+800//2, 330+30//2, 0.001], init_size=[800, 30])
        obstacle_2 = Obstacle(init_pos=[400//2, 530+30//2, 0.001], init_size=[400, 30])
        obstacle_3 = Obstacle(init_pos=[400+150+500//2, 530+30//2, 0.001], init_size=[500, 30])
        obstacle_4 = Obstacle(init_pos=[400+150+700, 750+30//2, -2.3], init_size=[600, 30])
        obstacle_5 = Obstacle(init_pos=[1000-30//2, 230//2+30, 0.001], init_size=[30, 230])
        # obstacle_6 = Obstacle(init_pos=[100, 500, 5])
        # obstacle_7 = Obstacle(init_pos=[500, 500, 0.001])
        # obstacle_8 = Obstacle(init_pos=[200, 200, 0.001])
        # obstacle_9 = Obstacle(init_pos=[500, 100, 1.571])
        # obstacle_10 = Obstacle(init_pos=[100, 500, 5])
        # obstacle_11 = Obstacle(init_pos=[500, 500, 0.001])

        self.add_obstacle(obstacle_l)
        self.add_obstacle(obstacle_r)
        self.add_obstacle(obstacle_t)
        self.add_obstacle(obstacle_b)

        self.add_obstacle(obstacle_0)
        self.add_obstacle(obstacle_1)
        self.add_obstacle(obstacle_2)
        self.add_obstacle(obstacle_3)
        self.add_obstacle(obstacle_4)
        self.add_obstacle(obstacle_5)

        # map.add_obstacle(obstacle_6)
        # map.add_obstacle(obstacle_7)
        # map.add_obstacle(obstacle_8)
        # map.add_obstacle(obstacle_9)
        # map.add_obstacle(obstacle_10)
        # map.add_obstacle(obstacle_11)

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def get_obstacles(self):
        return self.obstacles

    def generate(self):
        pass

   

    def update(self):
        # self.path = solve(self.resized_map, (0 , 0), (10 , 10))
        pass

    def draw(self, screen):
        for obstacle in self.obstacles:
            obstacle.draw(screen)
        # surf = pg.surfarray.make_surface(self.map)
        # screen.blit(surf, (0, 0))


