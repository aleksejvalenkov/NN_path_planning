import pygame as pg
import numpy as np
import sys
import os
import copy
# import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from gui.palette import *
from environment.collision import *
       

class Lidar2D:
    def __init__(self, transform) -> None:
        self.fov = 360 # in degree
        self.fov_rad = np.radians(self.fov) # in radians
        self.rays = 180 # pixels in 1d image from camera
        self.ray_lenght = 500
        self.rays_angles = np.arange(start=-self.fov_rad/2, stop=self.fov_rad/2, step=self.fov_rad/self.rays)
        self.rays_end_points_init = [np.array([self.ray_lenght * np.cos(rays_angle), self.ray_lenght * np.sin(rays_angle), 1]) for rays_angle in self.rays_angles]
        self.rays_end_points = [np.array([0, 0, 1]) for _ in range(len(self.rays_angles))]
        self.transform = transform
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        for i in range(len(self.rays_end_points)):
            self.rays_end_points[i] =  self.transform @ self.rays_end_points_init[i].T
        # print(self.rays_end_points)
        # print(self.transform)
        self.lidar_scan = []
        
    def get_ray_lines(self):
        edge_points_coordinates = []
        for point in self.rays_end_points:
            edge_points_coordinates.append([[self.x , self.y],[point[0], point[1]]])
        return edge_points_coordinates

    def update(self, transform):
        self.transform = transform
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        for i in range(len(self.rays_end_points)):
            self.rays_end_points[i] =  self.transform @ self.rays_end_points_init[i].T

    def scan(self, obstacles_obj):
        rays = self.get_ray_lines()
        obstacles = []
        for obstacle in obstacles_obj:
            obstacles.append(obstacle.get_edge_points())
        obstacles = list(obstacles)
        self.lidar_scan = self.scan_process(rays, obstacles)
        # print(type(self.lidar_scan), type(rays), type(obstacles))


        # print(self.lidar_scan)
        return self.lidar_scan

    def scan_process(self, rays, obstacles):
        # print(scan_new, rays, obstacles)
        scan_new = []
        for ray in rays:
            points = []
            point_min = copy.copy(ray[1])
            dist_min = np.inf
            for obstacle in obstacles:
                point = find_collision_rays(ray, obstacle)
                dist = distanse(ray[0], point)
                if dist <= dist_min:
                    dist_min = dist
                    point_min = point
            if dist_min <= self.ray_lenght-1:
                scan_new.append(point_min)
            else:
                scan_new.append([np.inf, np.inf])
        return scan_new

    def draw(self, screen):
        # draw sensor
        pg.draw.circle(screen, sensor_color, [self.x , self.y], 7, 3)
        # draw rays
        # for ray in self.rays_angles:
        #     pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(ray+self.theta) * self.ray_lenght , self.y + np.sin(ray+self.theta) * self.ray_lenght])
        for point in self.lidar_scan:
            # print('Отрисовка точки пересечения лидара', point[0] , point[1])
            if point[0] != np.inf or point[1] != np.inf:
                pg.draw.circle(screen, (255,0,0), [point[0] , point[1]], 3, 3)

        # draw SensorBar
        # screen_size = screen.get_size()
        # rect_size = 20
        # lens = copy.copy(self.matrix)
        # lens[lens >= self.ray_lenght] = self.ray_lenght
        # lens[:] = lens[:] * 255 / 500
        # # print(lens)
        # for i in range(len(lens)):
        #     r = round(lens[i])
        #     pg.draw.rect(screen, (r, r, r), 
        #          (rect_size*i, screen_size[1]-rect_size , rect_size, rect_size))