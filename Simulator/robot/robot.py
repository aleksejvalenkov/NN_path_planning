import pygame as pg
import numpy as np
import sys
import os

from planning.A_star import solve
from simple_pid import PID

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from robot.sensors import *
from utils.utils import *
from gui.palette import *
from environment.collision import *



import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn


class Robot:
    def __init__(self, map, init_pos=[100,100, 1]) -> None:

        self.length_m = 0.6 # In meter 
        self.width_m = 0.4 # In meter
        self.length_px = self.length_m * 100 # In pixels
        self.width_px = self.width_m * 100 # In pixels
        self.x , self.y, self.theta = init_pos # robot's center coordinate
        self.edge_points_init = [np.array([-self.length_px/2, -self.width_px/2, 1]),
                                 np.array([-self.length_px/2, +self.width_px/2, 1]),
                                 np.array([+self.length_px/2, +self.width_px/2, 1]),
                                 np.array([+self.length_px/2, -self.width_px/2, 1])]


        self.edge_points = [[],[],[],[]]

        self.robot_radius = 10
        self.throttle = 2

        self.bool_map = map
        
        self.t_vec = np.array([self.x , self.y])
        self.transform = get_transform(self.t_vec, self.theta)

        for i in range(len(self.edge_points)):
            point = self.transform @ self.edge_points_init[i].T
            self.edge_points[i] = [point[0], point[1]]
            # self.edge_points[i] = self.transform @ self.edge_points_init[i].T
        
        # print(self.edge_points)

        self.collision = False, None, None

        self.path = None
        self.way_point = (550//10, 550//10, 2)
        # print(f'Robot transform: \n {self.transform}')
        #Init Sensors
        self.lidar_transform = np.array( 
            [[  1., 0., self.robot_radius ],
             [  0., 1., 0. ],
             [  0., 0., 1. ]]
             )
        self.lidar = Lidar2D(self.transform @ self.lidar_transform)
        self.depth_image = None

        self.auto_mode = False
        self.pid_theta = PID(0.5, 0.0001, 0, 0)
        self.pid_theta.sample_time = 0.0033

        self.model = torch.load('/home/alex/Documents/NN_path_planning/nn/model/model_v2.pth')

        self.robot_points = []


    # def loop(self):

    #     while True:



    def update(self, map):

        for i in range(len(self.edge_points)):
            # self.edge_points[i] = self.transform @ self.edge_points_init[i].T
            point = self.transform @ self.edge_points_init[i].T
            self.edge_points[i] = [point[0], point[1]]

        obstacles = map.get_obstacles()
        
        for obstacle in obstacles:
            self.collision = find_collision(self.get_edge_points(), obstacle.get_edge_points())
            if self.collision[0]:
                break


        # self.cell_size = round(1/map.scale)

        lidar_transform = self.transform @ self.lidar_transform
        self.lidar.update(lidar_transform)
        # self.lidar_scan = []
        self.lidar_scan = self.lidar.scan(obstacles)

        # print(f'Robot pose on map {(self.x//10, self.y//10)}')

        self.robot_pose_on_map = (int(self.x//10), int(self.y//10))

        
        # print(self.get_waypoint_in_local())

        # self.action = self.nn_brain(self.get_observation(self.get_waypoint_in_local(), self.depth_image))
        # print(f'Action: {self.action}')

        if self.auto_mode:
            self.controll(self.action)
        

    def get_observation(self, goal, depth_image):
        obs = list(goal)
        obs.extend(list(depth_image))
        obs = np.array(obs)
        return obs
    
    def get_edge_points(self):
        return self.edge_points
    
    def nn_brain(self, obs):
        img_as_img = obs
        goal_np = img_as_img[:2]
        # goal_np = np.array([0.,0.])
        goal_np /= 1000.0
        img_as_img = img_as_img[2:]
        img_as_img /= 10000.0
        img_as_tensor = torch.from_numpy(img_as_img.astype('float32'))
        img_as_tensor = torch.unsqueeze(img_as_tensor, 0)
        img_as_tensor = torch.cat(tuple([img_as_tensor for item in range(len(img_as_img))]), 0)
        img_as_tensor = torch.unsqueeze(img_as_tensor, 0)
        img_as_tensor = torch.unsqueeze(img_as_tensor, 0)

        goal_as_tensor = torch.from_numpy(goal_np.astype('float32'))
        goal_as_tensor = torch.unsqueeze(goal_as_tensor, 1)
        goal_as_tensor = torch.unsqueeze(goal_as_tensor, 1)
        goal_as_tensor = torch.unsqueeze(goal_as_tensor, 0)

        # print(img_as_tensor.shape, goal_as_tensor.shape)
        output = self.model.forward(img_as_tensor, goal_as_tensor)
        # print(output)
        action = F.softmax(output).detach().numpy().argmax()
        return action

    def controll(self, action):

            # self.teleop(teleop_vec=[self.throttle,0,self.theta_speed])
            pass

    def teleop(self, teleop_vec):
        scale = 4
        self.t_vec = np.array([teleop_vec[0], teleop_vec[1]])
        # Pushing away vector

        move_vector = self.t_vec
        
        if self.collision[0]:
            collision_local = get_XY(inv(self.transform) @ get_transform(np.array([self.collision[1] , self.collision[2]]), 0))
            pushing_away_vector = np.array([collision_local[0] , collision_local[1]]) - np.array([0 , 0])

            # print('pushing_away_vector', pushing_away_vector)
            pushing_away_vector = pushing_away_vector / np.linalg.norm(pushing_away_vector)
            # print('pushing_away_vector NORM', pushing_away_vector)
            vector_coincidence = np.arccos(np.dot(self.t_vec, pushing_away_vector) / (np.linalg.norm(self.t_vec) * np.linalg.norm(pushing_away_vector))) # in radians
            if vector_coincidence <= 2.9:
                move_vector = self.t_vec - pushing_away_vector
        
            

        # print(move_vector)

        teleop_transform =  get_transform(move_vector * scale, teleop_vec[2])
        self.transform = self.transform @ teleop_transform
        
        # print(get_XYTheta(self.transform))
        self.x , self.y, self.theta = get_XYTheta(self.transform)

    def draw(self, screen):

        # pg.draw.circle(screen, robot_color, (self.x , self.y), self.robot_radius, 5)
        # pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(self.theta) * self.robot_radius , self.y + np.sin(self.theta) * self.robot_radius])
        # Drowing robot borders
        pg.draw.aaline(screen, robot_color, (self.edge_points[0][0], self.edge_points[0][1]), (self.edge_points[1][0], self.edge_points[1][1]))
        pg.draw.aaline(screen, robot_color, (self.edge_points[1][0], self.edge_points[1][1]), (self.edge_points[2][0], self.edge_points[2][1]))
        pg.draw.aaline(screen, robot_color, (self.edge_points[2][0], self.edge_points[2][1]), (self.edge_points[3][0], self.edge_points[3][1]))
        pg.draw.aaline(screen, robot_color, (self.edge_points[3][0], self.edge_points[3][1]), (self.edge_points[0][0], self.edge_points[0][1]))


        self.lidar.draw(screen)

        # c = self.cell_size
        # # pg.draw.rect(screen, robot_color, (self.robot_pose_on_map[0]*c, self.robot_pose_on_map[1]*c, c, c))
        # # Draw path
        # if self.path is not None:
        #     for cell in self.path:
        #         pg.draw.rect(screen, way_color, (cell[0]*c, cell[1]*c , c, c))
        if self.collision[0]:
            pg.draw.circle(screen, way_color, [self.collision[1] , self.collision[2]], 7, 3)

        

    def get_pose(self):
        return self.x , self.y, self.theta
    
    def get_waypoint_in_local(self):
        way_point = np.array(self.way_point) * 10
        way_point[2] = 1
        way_point_loc = np.linalg.inv(self.transform) @ way_point.T
        way_point_loc = way_point_loc[:2]
        # print(way_point_loc)
        return way_point_loc.T
    

    

    

    






