import pygame as pg
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from gui.palette import *
from environment.collision import *


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
    
    def update(self, obstacles):
        pass

    def draw(self, screen, color = border_color):

        # Drowing robot borders
        # print('points  ',self.edge_points)s
        pg.draw.polygon(screen, color, self.edge_points)
        # pg.draw.aaline(screen, robot_color, (self.edge_points[0][0], self.edge_points[0][1]), (self.edge_points[1][0], self.edge_points[1][1]))
        # pg.draw.aaline(screen, robot_color, (self.edge_points[1][0], self.edge_points[1][1]), (self.edge_points[2][0], self.edge_points[2][1]))
        # pg.draw.aaline(screen, robot_color, (self.edge_points[2][0], self.edge_points[2][1]), (self.edge_points[3][0], self.edge_points[3][1]))
        # pg.draw.aaline(screen, robot_color, (self.edge_points[3][0], self.edge_points[3][1]), (self.edge_points[0][0], self.edge_points[0][1]))


class MoveableObstacle:
    def __init__(self, init_pos) -> None:


        self.DRAW_TARGET = True

        self.fov = 360 # in degree
        self.fov_rad = np.radians(self.fov) # in radians
        self.rays = 10 # pixels in 1d image from camera
        self.radius = 25
        self.rays_angles = np.arange(start=-self.fov_rad/2, stop=self.fov_rad/2, step=self.fov_rad/self.rays)
        self.edge_points_init = [np.array([self.radius * np.cos(rays_angle), self.radius * np.sin(rays_angle), 1]) for rays_angle in self.rays_angles]

        self.x , self.y, self.theta = init_pos # robot's center coordinate
        # self.edge_points_init = [np.array([-self.length_px/2, -self.width_px/2, 1]),
        #                          np.array([-self.length_px/2, +self.width_px/2, 1]),
        #                          np.array([+self.length_px/2, +self.width_px/2, 1]),
        #                          np.array([+self.length_px/2, -self.width_px/2, 1])]

        self.edge_points = copy.copy(self.edge_points_init)
        self.n_steps = 0

        self.robot_radius = 10
        self.Vx = 0
        self.Vy = 0
        self.W = 0
        self.target = init_pos

        self.map = map
        
        self.t_vec = np.array([self.x , self.y])
        self.transform = get_transform(self.t_vec, self.theta)

        for i in range(len(self.edge_points)):
            point = self.transform @ self.edge_points_init[i].T
            self.edge_points[i] = [point[0], point[1]]
            # self.edge_points[i] = self.transform @ self.edge_points_init[i].T
        
        # print(self.edge_points)

        self.collision = False, None, None
        self.auto_mode = False
        self.trajectory = [np.array(self.get_pose())[0:2]]

        self.state = 0
        self.state_1_counter = 0
        

    def update(self, obstacles):
        if self.state == 0:
            self.teleop(teleop_vec=[0.1,0,0])
        if self.state == 1:
            self.teleop(teleop_vec=[-0.05,0,0.157])
            self.state_1_counter += 1

        if self.state_1_counter > 10:
            self.state = 0
            self.state_1_counter = 0

        self.n_steps += 1

        for i in range(len(self.edge_points)):
            point = self.transform @ self.edge_points_init[i].T
            self.edge_points[i] = [point[0], point[1]]

        obstacles_lines = []
        for obstacle in obstacles:
            obstacles_lines.extend(obstacle.get_lines())

        self.collision = find_collision(self.get_lines(), obstacles_lines)

        if self.collision[0]:
            self.state = 1

        
    def set_target(self, target):
        self.target = target
        self.Dt_l = np.linalg.norm(np.array(self.get_pose())[0:2] - np.array(self.target)[0:2])

    def get_observation(self, goal, depth_image):
        obs = list(goal)
        obs.extend(list(depth_image))
        obs = np.array(obs)
        return obs
    
    def get_edge_points(self):
        return self.edge_points
    
    def get_lines(self):
        points_1 = np.array(self.edge_points)
        points_2 = np.roll(points_1,1,axis=0)
        lines = np.stack((points_1, points_2), axis=1)
        return lines

    
 

    def teleop(self, teleop_vec):
        scale = 4
        self.t_vec = np.array([teleop_vec[0], teleop_vec[1]])
        # Pushing away vector

        move_vector = self.t_vec
        
        if self.collision[0]:
            collision_local = get_XY(inv(self.transform) @ get_transform(np.array([self.collision[1] , self.collision[2]]), 0))
            pushing_away_vector = np.array([collision_local[0] , collision_local[1]]) - np.array([0 , 0])

            # print('pushing_away_vector', pushing_away_vector)
            pushing_away_vector = (pushing_away_vector / np.linalg.norm(pushing_away_vector))
            # print('pushing_away_vector NORM', pushing_away_vector)
            vector_coincidence = np.arccos(np.dot(self.t_vec, pushing_away_vector) / (np.linalg.norm(self.t_vec) * np.linalg.norm(pushing_away_vector))) # in radians
            if vector_coincidence <= 2.9:
                move_vector = self.t_vec - pushing_away_vector
        
            
        # print(move_vector)

        teleop_transform =  get_transform(move_vector * scale, teleop_vec[2])
        self.transform = self.transform @ teleop_transform
        
        # print(get_XYTheta(self.transform))
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        self.trajectory.append(np.array(self.get_pose())[0:2])

    def draw(self, screen, color = green):

        # Drowing robot borders
        pg.draw.polygon(screen, color, self.edge_points)

        # draw trajectory
        # for i in range(1,len(self.trajectory)):
        #     self.trajectory[i]
        #     pg.draw.aaline(screen, robot_color, self.trajectory[i-1], self.trajectory[i])


        if self.collision[0]:
            pg.draw.circle(screen, way_color, [self.collision[1] , self.collision[2]], 7, 3)
  

    def get_pose(self):
        return self.x , self.y, self.theta
    

    