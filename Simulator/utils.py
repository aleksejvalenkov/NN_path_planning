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
    def __init__(self, transform) -> None:
        self.transform = transform
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        self.radius = 10

        pass

    def update(self):
        self.x , self.y, self.theta = get_XYTheta(self.transform)

    def draw(self, screen):
        pg.draw.circle(screen, sensor_color, (self.x , self.y), self.radius, 4)

    def get_pose(self):
        return [self.x, self.y]

class Robot:
    def __init__(self) -> None:
        self.robot_radius = 25
        self.x , self.y, self.theta = [100,100, 0]
        self.est_robot_pose = [0, 0, 0]
        self.t_vec = np.array([ self.x , self.y])
        self.transform = get_transform(self.t_vec, self.theta)
        self.way_point = (500, 500)


        self.robot_sensor_1_transform = np.array( 
            [[  1., 0., 0. ],
             [  0., 1., self.robot_radius ],
             [  0., 0., 1. ]]
             )
        self.robot_sensor_2_transform = np.array( 
            [[  1., 0., 0. ],
             [  0., 1., -self.robot_radius ],
             [  0., 0., 1. ]]
             )
        self.sensor_1_transform = np.array( 
            [[  1., 0., 500. ],
             [  0., 1., 20. ],
             [  0., 0., 1. ]]
             )
        self.sensor_2_transform = np.array( 
            [[  1., 0., 20. ],
             [  0., 1., 980. ],
             [  0., 0., 1. ]]
             )
        self.sensor_3_transform = np.array( 
            [[  1., 0., 960. ],
             [  0., 1., 960. ],
             [  0., 0., 1. ]]
             )

        #Init Sensors
        self.robot_sensor_1 = UWBSensor(self.transform @ self.robot_sensor_1_transform)
        self.robot_sensor_2 = UWBSensor(self.transform @ self.robot_sensor_2_transform)
        
        self.sensor_1 = UWBSensor(self.sensor_1_transform)
        self.sensor_2 = UWBSensor(self.sensor_2_transform)
        self.sensor_3 = UWBSensor(self.sensor_3_transform)

        self.throttle = 0.0
        self.theta_speed = 0.0
        self.auto_mode = False
        self.pid_throttle = PID(0.1, 0.0, 0, 0)
        self.pid_theta = PID(0.0001, 0.0, 0, 0)
        self.pid_throttle.sample_time = 0.0001
        self.pid_theta.sample_time = 0.0001


    def update(self):
        self.robot_sensor_1.transform = self.transform @ self.robot_sensor_1_transform
        self.robot_sensor_1.update()
        self.robot_sensor_2.transform = self.transform @ self.robot_sensor_2_transform
        self.robot_sensor_2.update()

        self.est_pose_sensor_1 = self.get_sensor_pose(self.robot_sensor_1)
        self.est_pose_sensor_2 = self.get_sensor_pose(self.robot_sensor_2)
        self.est_robot_pose = (self.est_pose_sensor_1[0] + self.est_pose_sensor_2[0])/2,\
              (self.est_pose_sensor_1[1] + self.est_pose_sensor_2[1])/2,\
                  -np.arctan2((self.est_pose_sensor_1[0] - self.est_pose_sensor_2[0]),(self.est_pose_sensor_1[1] - self.est_pose_sensor_2[1]))


        if self.auto_mode:
            self.pid_controll()

    def pid_controll(self):
        d = 4
        way_point = self.way_point
        if way_point[0] - d <= self.x <= way_point[0] + d and way_point[1] - d <= self.y <= way_point[1] + d :
            print('In goal')
        else:
            theta_setpoint = np.arctan2(-(self.est_robot_pose[1] - self.way_point[1]),-(self.est_robot_pose[0] - self.way_point[0]))
            self.pid_theta.setpoint = theta_setpoint
            self.pid_throttle.setpoint = 0

            if self.est_robot_pose[2] > math.pi - math.pi / 2 and theta_setpoint < 0:
                theta = (self.est_robot_pose[2] * -1) - math.pi / 2
            elif self.theta < -math.pi + math.pi / 2 and theta_setpoint > 0:
                theta = (self.est_robot_pose[2] * -1) + math.pi / 2
            else:
                theta = self.est_robot_pose[2]

            self.theta_speed = -constrain(self.pid_theta(theta), -0.1, 0.1)
            self.throttle = constrain(self.pid_throttle(-distance(self.get_pose()[:2], self.way_point)), 0, 4)
            # print(self.throttle, self.theta_speed)
            self.teleop([self.throttle, 0, self.theta_speed])

    def get_sensor_pose(self, sensor):
        self.d1 = distance(sensor.get_pose(), self.sensor_1.get_pose()) 
        self.d2 = distance(sensor.get_pose(), self.sensor_2.get_pose())
        self.d3 = distance(sensor.get_pose(), self.sensor_3.get_pose())

        self.p1 = point_form_two_rounds(self.sensor_1.get_pose(), self.d1, self.sensor_2.get_pose(), self.d2)
        self.p2 = point_form_two_rounds(self.sensor_1.get_pose(), self.d1, self.sensor_3.get_pose(), self.d3)
        self.p3 = point_form_two_rounds(self.sensor_2.get_pose(), self.d2, self.sensor_3.get_pose(), self.d3)

        V1 = line_from_two_point(self.sensor_1.get_pose(), self.sensor_2.get_pose())
        L1 = line_from_vec_and_point(V1[:2], self.p1)

        V2 = line_from_two_point(self.sensor_1.get_pose(), self.sensor_3.get_pose())
        L2 = line_from_vec_and_point(V2[:2], self.p2)

        V3 = line_from_two_point(self.sensor_2.get_pose(), self.sensor_3.get_pose())
        L3 = line_from_vec_and_point(V3[:2], self.p3)

        self.P1 = point_from_two_lines(L1, L2)
        self.P2 = point_from_two_lines(L1, L3)
        self.P3 = point_from_two_lines(L2, L3)

        pose = [(self.P1[0] + self.P2[0] + self.P3[0]) / 3 , (self.P1[1] + self.P2[1] + self.P3[1]) / 3]

        return np.array(pose)

    def teleop(self, teleop_vec):

        self.t_vec = np.array([ teleop_vec[0] , teleop_vec[1]])
        teleop_transform =  get_transform(self.t_vec, teleop_vec[2])
        self.transform = self.transform @ teleop_transform
        # print(get_XYTheta(self.transform))
        self.x , self.y, self.theta = get_XYTheta(self.transform)

    def draw(self, screen):

        pg.draw.circle(screen, robot_color, (self.x , self.y), self.robot_radius, 5)
        pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(self.theta) * self.robot_radius , self.y + np.sin(self.theta) * self.robot_radius])

        self.sensor_1.draw(screen)
        self.sensor_2.draw(screen)
        self.sensor_3.draw(screen)
        self.robot_sensor_1.draw(screen)
        self.robot_sensor_2.draw(screen)

        pg.draw.aaline(screen, sensor_color, self.robot_sensor_1.get_pose(), self.sensor_1.get_pose())
        pg.draw.aaline(screen, sensor_color, self.robot_sensor_1.get_pose(), self.sensor_2.get_pose())
        pg.draw.aaline(screen, sensor_color, self.robot_sensor_1.get_pose(), self.sensor_3.get_pose())

        pg.draw.aaline(screen, sensor_color, self.robot_sensor_2.get_pose(), self.sensor_1.get_pose())
        pg.draw.aaline(screen, sensor_color, self.robot_sensor_2.get_pose(), self.sensor_2.get_pose())
        pg.draw.aaline(screen, sensor_color, self.robot_sensor_2.get_pose(), self.sensor_3.get_pose())

        pg.draw.circle(screen, (255,0,0), self.est_pose_sensor_1, 5, 5)
        pg.draw.circle(screen, (0,0,255), self.est_pose_sensor_2, 5, 5)

        pg.draw.circle(screen, (100,100,255), self.way_point, 5, 5)

    def get_pose(self):
        return self.x , self.y, self.theta
    

