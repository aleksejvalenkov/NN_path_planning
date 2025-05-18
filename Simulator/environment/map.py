import pygame as pg
import numpy as np
import sys
import os
import random

import copy

import cv2 


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from environment.obstacles import *

def distance(point_1, point_2):
    return np.sqrt(np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]))

class Map:
    def __init__(self, size = (800, 800), seed=None, robot_pose=None) -> None:
        self.global_map = None
        if seed is not None:
            random.seed(seed)
        self.obs_param = 200
        self.size = size
        self.scale = 0.1
        self.expansion_cells = 3
        self.obstacles = []
        self.moveable_obstacles = []
        # Cтены 
        # Нужен генератор карт
        border_width = 10
        if robot_pose is None:
            self.robot_pose  = [100,100,0]
        else:
            self.robot_pose = robot_pose

        obstacle_l = Obstacle(init_pos=[border_width//2, size[1]//2, 0], init_size=[border_width, size[1]])
        obstacle_r = Obstacle(init_pos=[size[0]-border_width//2, size[1]//2, 0], init_size=[border_width, size[1]])
        obstacle_t = Obstacle(init_pos=[size[0]//2, border_width//2, 0], init_size=[size[0], border_width])
        obstacle_b = Obstacle(init_pos=[size[0]//2, size[1]-border_width//2, 0], init_size=[size[0], border_width])

        obstacle_0 = Obstacle(init_pos=[200, 100, 0], init_size=[30, 200])
        obstacle_1 = Obstacle(init_pos=[1400, 300, 0], init_size=[30, 200])
        obstacle_2 = Obstacle(init_pos=[650, 350, 0], init_size=[30, 100])
        obstacle_3 = Obstacle(init_pos=[600, 220, -0.785], init_size=[30, 200])
        obstacle_4 = Obstacle(init_pos=[1000, 120, 0.785], init_size=[30, 300])
        obstacle_5 = Obstacle(init_pos=[1000, 120, -0.785], init_size=[30, 300])

        obstacle_box_1 = Obstacle(init_pos=[350, 350, 0], init_size=[30, 30])
        obstacle_box_2 = Obstacle(init_pos=[300, 400, 0], init_size=[80, 80])
        obstacle_box_3 = Obstacle(init_pos=[500, 400, 0], init_size=[80, 80])
        obstacle_box_4 = Obstacle(init_pos=[620, 620, 1], init_size=[200, 50])
        obstacle_box_5 = Obstacle(init_pos=[600, 150, 0], init_size=[80, 80])

        # obstacle_7 = Obstacle(init_pos=[500, 500, 0])
        # obstacle_8 = Obstacle(init_pos=[200, 200, 0])
        # obstacle_9 = Obstacle(init_pos=[500, 100, 1.571])
        # obstacle_10 = Obstacle(init_pos=[100, 500, 5])
        # obstacle_11 = Obstacle(init_pos=[500, 500, 0])

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

        self.add_obstacle(obstacle_box_1)
        # self.add_obstacle(obstacle_box_2)
        # self.add_obstacle(obstacle_box_3)
        # self.add_obstacle(obstacle_box_4)
        # self.add_obstacle(obstacle_box_5)
        # map.add_obstacle(obstacle_7)
        # map.add_obstacle(obstacle_8)
        # map.add_obstacle(obstacle_9)
        # map.add_obstacle(obstacle_10)
        # map.add_obstacle(obstacle_11)

        self.bin_map, self.bin_map_og = self.init_bool_map()

        self.bin_map_og_rgb = cv2.imread("global_planner/map/map_bin_ext.jpg")

        for i in range(15):
            moveable_obstacle = MoveableObstacle(init_pos=self.get_random_pose())
            self.moveable_obstacles.append(moveable_obstacle)


    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def get_obstacles(self):
        obstacles = self.obstacles + self.moveable_obstacles
        return obstacles
    
    def get_static_obstacles(self):
        obstacles = self.obstacles
        return obstacles
    
    def get_moveable_obstacles(self):
        obstacles = self.moveable_obstacles
        return obstacles


    def get_random_pose(self):
        while True:
            x = random.randint(0, self.size[0]-1)
            y = random.randint(0, self.size[1]-1)
            if not self.bin_map_og[y,x] and distance([x,y], [self.robot_pose[0], self.robot_pose[1]]) > 100:
                    break

        fi = (random.random()-0.5)*2*np.pi
        return x,y,fi

    def generate(self):
        pass

    def set_global_map(self, global_map):
        self.global_map = global_map

    def init_bool_map(self, blur=100):
        pg.init()
        surface = pg.Surface(self.size)
        for obstacle in self.obstacles:
            obstacle.draw(surface, color = (255,255,255))
        # pg.image.save(surface, "global_planner/map/map_bin.jpg")

        map = cv2.imread("global_planner/map/map_bin.jpg")
        # print(map)
        gray_map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((blur,blur),np.float32)/25
        gray_map_blur = cv2.filter2D(gray_map,-1,kernel)
        gray_map_bin_ext = np.where(gray_map_blur > 127, 1, 0)
        gray_map_bin = np.where(gray_map > 127, 1, 0)

        # cv2.imwrite("map_bin_ext.jpg", gray_map_bin_ext*255)
        return gray_map_bin, gray_map_bin_ext

    def update(self, render_fps):
        # self.path = solve(self.resized_map, (0 , 0), (10 , 10))
        for obstacle in self.moveable_obstacles:
            obstacle.update(self.obstacles, render_fps)


    def draw(self, screen):
        for obstacle in self.obstacles:
            obstacle.draw(screen)

        for obstacle in self.moveable_obstacles:
            obstacle.draw(screen)



        # surf = pg.surfarray.make_surface(self.map)
        # screen.blit(surf, (0, 0))


