import pygame as pg
import numpy as np
from numpy import floor
from transforms import *
from perlin_numpy import generate_perlin_noise_2d
import cv2 
import copy
import math
from planning.A_star import solve
from simple_pid import PID

map_color = (100,100,100)
map_color_2 = (150,150,150)
robot_color = (0,0,190)
sensor_color = (0,190,0)
silver = (194, 194, 194)
way_color = (200, 50, 50)


class UWBSensor:
    def __init__(self) -> None:
        pass

class Robot:
    def __init__(self) -> None:
        self.robot_radius = 25
        self.x , self.y = [100,100]

        #Init Sensors

        self.way_points = [[300,100],[300,300],[100,300],[100,100]]
        self.p = 0

        self.pid_x = PID(0.11, 0.0001, 0, 0)
        self.pid_y = PID(0.11, 0.0001, 0, 0)
        self.pid_x.sample_time = 0.033
        self.pid_y.sample_time = 0.033

    def update(self):
        self.pid_controll()
        pass
    
    def pid_controll(self):
        d = 10
        if self.way_points:
            way_point = self.way_points[0]
            if way_point[0] - d <= self.x <= way_point[0] + d and way_point[1] - d <= self.y <= way_point[1] + d :
                if len(self.way_points) >= 1:
                    self.way_points.pop(0)

                print('In goal')
                
            self.pid_x.setpoint = way_point[0]
            self.pid_y.setpoint = way_point[1]

            self.x_speed = constrain(self.pid_x(self.x), -1, 1)
            self.y_speed = constrain(self.pid_y(self.y), -1, 1)

            self.teleop([self.x_speed, self.y_speed ])

        

    def teleop(self, teleop_vec):

        self.x = self.x + teleop_vec[0]
        self.y = self.y + teleop_vec[1]

    def draw(self, screen):

        pg.draw.circle(screen, robot_color, (self.x , self.y), self.robot_radius, 5)
        # pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(self.theta) * self.robot_radius , self.y + np.sin(self.theta) * self.robot_radius])

    def get_pose(self):
        return self.x , self.y
    

