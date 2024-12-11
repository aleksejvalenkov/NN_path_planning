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
    def __init__(self, init_pos=[200,200, 1]) -> None:

        self.length_m = 2.4 # In meter 
        self.width_m = 0.3 # In meter
        self.length_px = self.length_m * 100 # In pixels
        self.width_px = self.width_m * 100 # In pixels
        self.x , self.y, self.theta = init_pos # Obstacle's center coordinate
        self.edge_points_init = [get_transform(np.array([-self.length_px/2, -self.width_px/2]), 0),
                            get_transform(np.array([-self.length_px/2, +self.width_px/2]), 0),
                            get_transform(np.array([+self.length_px/2, +self.width_px/2]), 0),
                            get_transform(np.array([+self.length_px/2, -self.width_px/2]), 0)]


        self.edge_points = [[],[],[],[]]

        self.robot_radius = 25
        self.throttle = 2
        
        self.t_vec = np.array([self.x , self.y])
        self.transform = get_transform(self.t_vec, self.theta)
        for j in range(456):
            for i in range(len(self.edge_points)):
                self.edge_points[i] = self.transform @ self.edge_points_init[i]

    def get_edge_points(self):
        edge_points_coordinates = []
        for i in range(len(self.edge_points)):
            edge_points_coordinates.append(get_XY(self.edge_points[i]))
        return edge_points_coordinates

    def draw(self, screen):

        # Drowing robot borders
        pg.draw.aaline(screen, robot_color, get_XY(self.edge_points[0]), get_XY(self.edge_points[1]))
        pg.draw.aaline(screen, robot_color, get_XY(self.edge_points[1]), get_XY(self.edge_points[2]))
        pg.draw.aaline(screen, robot_color, get_XY(self.edge_points[2]), get_XY(self.edge_points[3]))
        pg.draw.aaline(screen, robot_color, get_XY(self.edge_points[3]), get_XY(self.edge_points[0]))