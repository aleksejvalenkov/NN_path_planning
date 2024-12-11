import pygame as pg
import numpy as np
import sys
import os
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from gui.palette import *


class DepthSensor:
    def __init__(self, transform) -> None:
        self.fov = 150 # in degree
        self.fov_rad = np.radians(self.fov) # in radians
        self.pix = 40 # pixels in 1d image from camera
        self.transform = transform
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        self.rays_angles = np.arange(start=-self.fov_rad/2, stop=self.fov_rad/2, step=self.fov_rad/self.pix)
        self.ray_lenght = 500
        self.matrix = np.zeros(self.pix)
        self.range_points = np.zeros((self.pix, 2))

    def update(self):
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        # print(self.matrix)

    def scan(self, bool_map):
        for i_ray in range(len(self.rays_angles)):
            loc_points = []
            k = np.tan(self.rays_angles[i_ray])
            for x in range(0, self.ray_lenght, 2):
                y = k*x
                loc_points.append(np.array([x,y,1]).T)
            
            loc_points = np.array(loc_points)

            for loc_point in loc_points:
                point = self.transform @ loc_point
                x, y, _ = point
                x = round(x)
                y = round(y)
                x_size = len(bool_map) - 1
                y_size = len(bool_map[0]) - 1
                if 0 <= x <= x_size and 0 <= y <= y_size:
                    if bool_map[x][y] == 1:
                        self.range_points[i_ray] = (x,y)
                        break
                    else:
                        self.range_points[i_ray] = (np.inf, np.inf)

            self.matrix[i_ray] = np.sqrt(pow((self.x - self.range_points[i_ray][0]), 2) + pow((self.y - self.range_points[i_ray][1]), 2))
            self.matrix[self.matrix >= 10000] = 10000
        return self.matrix

    def draw(self, screen):
        # draw sensor
        pg.draw.circle(screen, sensor_color, [self.x , self.y], 7, 3)
        # draw rays
        for ray in self.rays_angles:
            pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(ray+self.theta) * self.ray_lenght , self.y + np.sin(ray+self.theta) * self.ray_lenght])
        for point in self.range_points:
            pg.draw.circle(screen, (255,0,0), [point[0] , point[1]], 3, 3)

        # draw SensorBar
        screen_size = screen.get_size()
        rect_size = 20
        lens = copy.copy(self.matrix)
        lens[lens >= self.ray_lenght] = self.ray_lenght
        lens[:] = lens[:] * 255 / 500
        # print(lens)
        for i in range(len(lens)):
            r = round(lens[i])
            pg.draw.rect(screen, (r, r, r), 
                 (rect_size*i, screen_size[1]-rect_size , rect_size, rect_size))
            

class Lidar2D:
    def __init__(self, transform) -> None:
        self.fov = 360 # in degree
        self.fov_rad = np.radians(self.fov) # in radians
        self.pix = 180 # pixels in 1d image from camera
        self.transform = transform
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        self.rays_angles = np.arange(start=-self.fov_rad/2, stop=self.fov_rad/2, step=self.fov_rad/self.pix)
        self.ray_lenght = 500
        self.matrix = np.zeros(self.pix)
        self.range_points = np.zeros((self.pix, 2))

    def update(self):
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        # print(self.matrix)

    def scan(self, bool_map):
        for i_ray in range(len(self.rays_angles)):
            loc_points = []
            k = np.tan(self.rays_angles[i_ray])
            for x in range(0, self.ray_lenght, 2):
                y = k*x
                loc_points.append(np.array([x,y,1]).T)
            
            loc_points = np.array(loc_points)

            for loc_point in loc_points:
                point = self.transform @ loc_point
                x, y, _ = point
                x = round(x)
                y = round(y)
                x_size = len(bool_map) - 1
                y_size = len(bool_map[0]) - 1
                if 0 <= x <= x_size and 0 <= y <= y_size:
                    if bool_map[x][y] == 1:
                        self.range_points[i_ray] = (x,y)
                        break
                    else:
                        self.range_points[i_ray] = (np.inf, np.inf)

            self.matrix[i_ray] = np.sqrt(pow((self.x - self.range_points[i_ray][0]), 2) + pow((self.y - self.range_points[i_ray][1]), 2))
            self.matrix[self.matrix >= 10000] = 10000
        return self.matrix

    def draw(self, screen):
        # draw sensor
        pg.draw.circle(screen, sensor_color, [self.x , self.y], 7, 3)
        # draw rays
        # for ray in self.rays_angles:
        #     pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(ray+self.theta) * self.ray_lenght , self.y + np.sin(ray+self.theta) * self.ray_lenght])
        for point in self.range_points:
            pg.draw.circle(screen, (255,0,0), [point[0] , point[1]], 3, 3)

        # draw SensorBar
        screen_size = screen.get_size()
        rect_size = 20
        lens = copy.copy(self.matrix)
        lens[lens >= self.ray_lenght] = self.ray_lenght
        lens[:] = lens[:] * 255 / 500
        # print(lens)
        for i in range(len(lens)):
            r = round(lens[i])
            pg.draw.rect(screen, (r, r, r), 
                 (rect_size*i, screen_size[1]-rect_size , rect_size, rect_size))