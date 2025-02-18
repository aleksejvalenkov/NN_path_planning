import pygame as pg
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from utils.utils import *
from gui.palette import *


class Obstacle:
    def __init__(self, init_pos=[200,200, 1], init_size=[240, 30]) -> None:

        # self.length_m = init_size[0] # In meter 
        # self.width_m = init_size[1] # In meter
        self.length_px = init_size[0] # In pixels
        self.width_px = init_size[1] # In pixels
        self.x , self.y, self.theta = init_pos # Obstacle's center coordinate
        self.edge_points_init = [np.array([-self.length_px/2, -self.width_px/2, 1]),
                                 np.array([-self.length_px/2, +self.width_px/2, 1]),
                                 np.array([+self.length_px/2, +self.width_px/2, 1]),
                                 np.array([+self.length_px/2, -self.width_px/2, 1])]


        self.edge_points = [[],[],[],[]]

        self.robot_radius = 25
        self.throttle = 2
        
        self.t_vec = np.array([self.x , self.y])
        self.transform = get_transform(self.t_vec, self.theta)
        for i in range(len(self.edge_points)):
            point = self.transform @ self.edge_points_init[i].T
            self.edge_points[i] = [point[0], point[1]]

    def get_edge_points(self):
        # edge_points_coordinates = []
        # for point in self.edge_points:
        #     edge_points_coordinates.append([point[0], point[1]])
        # return edge_points_coordinates
        return self.edge_points
    
    def get_lines(self):
        return [self.edge_points[0], self.edge_points[1]], [self.edge_points[1], self.edge_points[2]], [self.edge_points[2], self.edge_points[3]], [self.edge_points[3], self.edge_points[0]]

    def draw(self, screen):

        # Drowing robot borders
        # print('points  ',self.edge_points)s
        pg.draw.polygon(screen, border_color, self.edge_points)
        # pg.draw.aaline(screen, robot_color, (self.edge_points[0][0], self.edge_points[0][1]), (self.edge_points[1][0], self.edge_points[1][1]))
        # pg.draw.aaline(screen, robot_color, (self.edge_points[1][0], self.edge_points[1][1]), (self.edge_points[2][0], self.edge_points[2][1]))
        # pg.draw.aaline(screen, robot_color, (self.edge_points[2][0], self.edge_points[2][1]), (self.edge_points[3][0], self.edge_points[3][1]))
        # pg.draw.aaline(screen, robot_color, (self.edge_points[3][0], self.edge_points[3][1]), (self.edge_points[0][0], self.edge_points[0][1]))