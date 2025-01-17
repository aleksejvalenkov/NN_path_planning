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


class Robot:
    def __init__(self, map, init_pos) -> None:

        self.length_m = 0.6 # In meter 
        self.width_m = 0.4 # In meter
        self.length_px = self.length_m * 100 # In pixels
        self.width_px = self.width_m * 100 # In pixels
        self.x , self.y, self.theta = init_pos # robot's center coordinate
        self.edge_points_init = [np.array([-self.length_px/2, -self.width_px/2, 1]),
                                 np.array([-self.length_px/2, +self.width_px/2, 1]),
                                 np.array([+self.length_px/2, +self.width_px/2, 1]),
                                 np.array([+self.length_px/2, -self.width_px/2, 1])]
        lin = 0.5
        ang = 0.05
        self.action_ves = {0:[0, 0], 1:[0,ang], 2:[lin,ang], 3:[lin,0], 4:[lin,-ang], 5:[0,-ang]}
        self.edge_points = [[],[],[],[]]
        self.n_steps = 0

        self.robot_radius = 10
        self.throttle = 0
        self.theta_speed = 0
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

        #Init Sensors
        self.lidar_transform = np.array( 
            [[  1., 0., self.robot_radius ],
             [  0., 1., 0. ],
             [  0., 0., 1. ]]
             )
        self.lidar = Lidar2D(self.transform @ self.lidar_transform)
        self.lidar_points = []
        self.lidar_distances = []

        self.auto_mode = False
        self.pid_theta = PID(0.5, 0.0001, 0, 0)
        self.pid_theta.sample_time = 0.0033

        self.state = np.zeros((16))
        self.update(self.map)
        self.get_state()
        self.Dt_l = np.linalg.norm(np.array(self.get_pose())[0:2] - np.array(self.target)[0:2])
        self.Xt_l = np.min(self.state[0:10])
        self.trajectory = [np.array(self.get_pose())[0:2]]


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

        self.lidar_points, self.lidar_distances = self.lidar.scan(obstacles)

        self.robot_pose_on_map = (int(self.x//10), int(self.y//10))

        
        # print(self.get_waypoint_in_local())


        if self.auto_mode:
            self.controll(self.action)
        
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

    def get_state(self):
        state = np.zeros((16))
        # 10 mins ranges in 30 rays of lidar [10]
        lidar_distances_norm = np.array(self.lidar_distances)/self.lidar.ray_lenght
        lidar_distances_mins = np.zeros((10))
        c = len(lidar_distances_norm) // len(lidar_distances_mins)
        for i in range(len(lidar_distances_mins)):
            lidar_distances_mins[i] = np.min(lidar_distances_norm[i*c:i*c+c])
        
        # Velocity lin, angular [2]
        velocity = np.array([np.exp(self.throttle), np.tanh(self.theta_speed)])
        # Target point vector [2]
        target_point_loc = inv(self.transform) @ np.array([self.target[0], self.target[1], 1])
        target_point_vector = np.array([target_point_loc[0], target_point_loc[1]])
        # robot orientation [1]
        robot_orientation = np.tanh(self.theta)
        # target orientation [1]
        target_orientation = np.tanh(self.target[2])

        state[0:10] = lidar_distances_mins
        state[10:12] = velocity
        state[12:14] = target_point_vector
        # print('target_point_vector ', target_point_vector)
        state[14] = robot_orientation
        state[15] = target_orientation
        # print(state.shape)
        # print(state)
        self.state = state
        return np.array(state)
    
    def get_reward(self):
        reward = 0
        terminated = False
        truncated = False
        max_revard = 1000
        Cd = 30
        Dt = np.linalg.norm(np.array(self.get_pose())[0:2] - np.array(self.target)[0:2])
        Co = 30/self.lidar.ray_lenght
        Cop = 100/self.lidar.ray_lenght
        Xt = np.min(self.state[0:10])
        max_steps = 1000
        # print('Xt= ', Xt)
        hd = self.state[15] - self.state[14]
        Cr = 10
        Cp = 0.0
        Cro = 5 * self.lidar.ray_lenght
        # print(self.Dt_l, Dt)
        if Dt < Cd :
            reward = max_revard
            truncated = True
        elif Xt < Co:
            reward = -100
            terminated = True
        elif self.n_steps > max_steps:
            reward = -100
            terminated = True
        # elif Xt < Cop:
        #     reward = Cr * (self.Dt_l - Dt) * pow(2,(self.Dt_l/Dt)) - Cp * (1 - hd) - Cro * (self.Xt_l - Xt) * pow(2,(self.Xt_l/Xt))
        else:
            reward = Cr * (self.Dt_l - Dt) * pow(2,(self.Dt_l/Dt)) - Cp * (1 - hd)
        # if reward < 0:
        #     reward = 0
        self.Dt_l = Dt
        self.Xt_l = Xt
        # print(reward)
        return reward, terminated, truncated


    def controll(self, action):
        # print(action)
        self.throttle, self.theta_speed = self.action_ves.get(action)
        self.teleop(teleop_vec=[self.throttle,0,self.theta_speed])
        self.n_steps += 1
        

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
        self.trajectory.append(np.array(self.get_pose())[0:2])

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

        # draw trajectory
        for i in range(1,len(self.trajectory)):
            self.trajectory[i]
            pg.draw.aaline(screen, robot_color, self.trajectory[i-1], self.trajectory[i])


        if self.collision[0]:
            pg.draw.circle(screen, way_color, [self.collision[1] , self.collision[2]], 7, 3)

        pg.draw.circle(screen, way_color, [self.target[0] , self.target[1]], 7, 3)

        

    def get_pose(self):
        return self.x , self.y, self.theta
    
    def get_waypoint_in_local(self):
        way_point = np.array(self.way_point) * 10
        way_point[2] = 1
        way_point_loc = np.linalg.inv(self.transform) @ way_point.T
        way_point_loc = way_point_loc[:2]
        # print(way_point_loc)
        return way_point_loc.T
    

    

    

    






