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
sensor_color_2 = (10,10,10)
silver = (194, 194, 194)
way_color = (200, 50, 50)


class UWBSensor:
    def __init__(self, pose) -> None:
        self.x, self.y = pose
        self.radius = 10

        pass

    def update(self):
        pass

    def draw(self, screen):
        pg.draw.circle(screen, sensor_color, (self.x , self.y), self.radius, 4)
        # pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(self.theta) * self.robot_radius , self.y + np.sin(self.theta) * self.robot_radius])

    def get_pose(self):
        return [self.x, self.y]

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

        self.sensor_1 = UWBSensor([20,980])
        self.sensor_2 = UWBSensor([500,20])
        self.sensor_3 = UWBSensor([980,980])

    def update(self):
        # self.pid_controll()
        self.est_pose =  self.get_robot_pose()
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

    def get_robot_pose(self):
        self.d1 = distance(self.get_pose(), self.sensor_1.get_pose()) 
        self.d2 = distance(self.get_pose(), self.sensor_2.get_pose())
        self.d3 = distance(self.get_pose(), self.sensor_3.get_pose())
        # print(self.d1, self.d2, self.d3)

        self.p1 = point_form_two_rounds(self.sensor_1.get_pose(), self.d1, self.sensor_2.get_pose(), self.d2)
        self.p2 = point_form_two_rounds(self.sensor_1.get_pose(), self.d1, self.sensor_3.get_pose(), self.d3)
        self.p3 = point_form_two_rounds(self.sensor_2.get_pose(), self.d2, self.sensor_3.get_pose(), self.d3)
        # print(self.p1)

        x = (self.p1[0] + self.p2[0] +self.p3[0]) / 3
        y = (self.p1[1] + self.p2[1] +self.p3[1]) / 3
        # x, y = hxy(self.p1, self.p2, self.p3)
        return [x, y]

    def teleop(self, teleop_vec):

        self.x = self.x + teleop_vec[0]
        self.y = self.y + teleop_vec[1]

    def draw(self, screen):

        pg.draw.circle(screen, robot_color, (self.x , self.y), self.robot_radius, 5)
        # pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(self.theta) * self.robot_radius , self.y + np.sin(self.theta) * self.robot_radius])

        self.sensor_1.draw(screen)
        self.sensor_2.draw(screen)
        self.sensor_3.draw(screen)

        pg.draw.circle(screen, sensor_color_2, self.sensor_1.get_pose(), self.d1, 1)
        pg.draw.aaline(screen, sensor_color, self.get_pose(), self.sensor_1.get_pose())

        pg.draw.circle(screen, sensor_color_2, self.sensor_2.get_pose(), self.d2, 1)
        pg.draw.aaline(screen, sensor_color, self.get_pose(), self.sensor_2.get_pose())

        pg.draw.circle(screen, sensor_color_2, self.sensor_3.get_pose(), self.d3, 1)
        pg.draw.aaline(screen, sensor_color, self.get_pose(), self.sensor_3.get_pose())

        pg.draw.circle(screen, sensor_color_2, self.p1, 5)
        pg.draw.circle(screen, sensor_color_2, self.p2, 5)
        pg.draw.circle(screen, sensor_color_2, self.p3, 5)

        pg.draw.circle(screen, way_color, self.est_pose, 5, 5)

    def get_pose(self):
        return [self.x , self.y]
    

