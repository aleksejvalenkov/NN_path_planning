import pygame as pg
import numpy as np
import sys
import os
import time

from planning.A_star import solve
from simple_pid import PID

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.transforms import *
from robot.sensors import *
from gui.palette import *
from environment.collision import *

METRIC_KF = 100 # 1m = 100px

class Robot:
    def __init__(self, map, init_pos) -> None:
        self.length_m = 0.51 # In meter 
        self.width_m = 0.32 # In meter
        self.length_wheel = 0.51 # In meter 
        self.width_wheel = 0.32 # In meter 
        self.wheel_radius = 0.1 # In meter 
        self.length_px = self.length_m * 100 # In pixels
        self.width_px = self.width_m * 100 # In pixels
        self.render_fps = 0
        self.joit_vec_ust = np.zeros((4))
        self.joit_ang_vel = np.zeros((4))

        self.DRAW_TARGET = True

        self.x , self.y, self.theta = init_pos # robot's center coordinate
        self.edge_points_init = [np.array([-self.length_px/2, -self.width_px/2, 1]),
                                 np.array([-self.length_px/2, +self.width_px/2, 1]),
                                 np.array([+self.length_px/2, +self.width_px/2, 1]),
                                 np.array([+self.length_px/2, -self.width_px/2, 1])]
        lin = 0.5
        ang = 0.05
        self.action_ves = {0:[0, 0, 0], 1:[0, 0,ang], 2:[lin, 0,ang], 3:[lin, 0,0], 4:[lin, 0,-ang], 5:[0, 0,-ang]}
        self.edge_points = [[],[],[],[]]
        self.n_steps = 0
        self.live_secs_start = time.time()
        self.live_secs = time.time() - self.live_secs_start

        self.robot_radius = 10
        self.Vx = 0 # In m/s
        self.Vy = 0 # In m/s 
        self.W = 0 # In rad/s
        self.max_vx = 1.0 # In m/s
        self.max_vy = 0.6 # In m/s
        self.max_w = 1.0 # In rad/s
        self.wheel_eps = 30 # angular acceleration for the wheel In 
        self.max_wheel_vel = 20

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

        self.trajectory = [np.array(self.get_pose())[0:2]]

        self.state = np.zeros((16))
        self.update(self.map, render_fps=0)
        self.get_state()
        self.Dt_l = np.linalg.norm(np.array(self.get_pose())[0:2] - np.array(self.target)[0:2])
        self.Xt_l = np.min(self.state[0:10])
        


    def update(self, map, render_fps):
        self.render_fps = render_fps
        self.live_secs = time.time() - self.live_secs_start
        self.integrate_ang_vel(self.joit_vec_ust)
        move_vec_real = self.joit_vec_to_move_vec(self.joit_ang_vel)
        # print('move_vec_real =', move_vec_real)
        self.teleop(teleop_vec=move_vec_real)
        self.n_steps += 1

        for i in range(len(self.edge_points)):
            # self.edge_points[i] = self.transform @ self.edge_points_init[i].T
            point = self.transform @ self.edge_points_init[i].T
            self.edge_points[i] = [point[0], point[1]]

        obstacles = map.get_obstacles()
        obstacles_lines = []
        for obstacle in obstacles:
            obstacles_lines.extend(obstacle.get_lines())

        self.collision = find_collision(self.get_lines(), obstacles_lines)

        # self.cell_size = round(1/map.scale)

        lidar_transform = self.transform @ self.lidar_transform
        self.lidar.update(lidar_transform)
        self.lidar_points, self.lidar_distances = self.lidar.scan(obstacles_lines)

        
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
        return [self.edge_points[0], self.edge_points[1]], [self.edge_points[1], self.edge_points[2]], [self.edge_points[2], self.edge_points[3]], [self.edge_points[3], self.edge_points[0]]

    def get_state(self):
        state = np.zeros((26))
        # 20 mins ranges in 60 rays of lidar [20]
        lidar_distances_norm = np.array(self.lidar_distances)/self.lidar.ray_lenght
        lidar_distances_mins = np.zeros((20))
        c = len(lidar_distances_norm) // len(lidar_distances_mins)
        for i in range(len(lidar_distances_mins)):
            lidar_distances_mins[i] = np.min(lidar_distances_norm[i*c:i*c+c])
        
        # Velocity lin, angular [2]
        velocity = np.array([np.tanh(self.Vx), np.tanh(self.Vy), np.tanh(self.W)])
        # Target point vector [2]
        target_point_loc = inv(self.transform) @ np.array([self.target[0], self.target[1], 1])
        target_point_vector = np.array([target_point_loc[0], target_point_loc[1]])
        # robot orientation [1]
        robot_orientation = self.theta
        # target orientation [1]
        target_orientation = self.target[2]
        
        state[0:20] = lidar_distances_mins # normalize from 0-500 to 0-1
        state[20:23] = velocity # in meters per sec
        state[23:25] = target_point_vector / METRIC_KF # in meters
        # print('target_point_vector ', target_point_vector)
        state[25] = get_theta_error(target_orientation, robot_orientation) # in rad
        # print(state.shape)
        # print(state)
        self.state = state
        return np.array(state)
    
    def get_reward(self):
        def positive(val):
            if val > 0:
                return val
            else:
                return 0
        reward = 0
        terminated = False
        truncated = False
        max_revard = 2000
        Cd = 10
        Dt = np.linalg.norm(np.array(self.get_pose())[0:2] - np.array(self.target)[0:2])
        Co = 0.3/self.lidar.ray_lenght
        Cop = 0.5/self.lidar.ray_lenght # in meter mast be < self.lidar.ray_lenght
        Xt = np.min(self.state[0:20])
        max_live_time = 20
        # print('Xt= ', Xt)
        hd = np.abs(self.state[25])
        Cr = 0.0
        Cp = 0.0
        Cro = 5.0 * self.lidar.ray_lenght * METRIC_KF
        # print('Dt = ', Dt)
        if Dt < Cd:
            reward = 3000 #* (1 - (self.n_steps / max_steps))
            truncated = True
        elif self.collision[0]: # Xt < Co or
            reward = -3000
            terminated = True
        elif self.live_secs > max_live_time:
            reward = -1000
            terminated = True
        elif Xt < Cop:
            reward = Cr * (self.Dt_l - Dt) * pow(2,(self.Dt_l/Dt)) - Cp * hd - Cro * positive((self.Xt_l - Xt) * pow(2,(self.Xt_l/Xt)))
        else:
            reward = Cr * (self.Dt_l - Dt) * pow(2,(self.Dt_l/Dt)) - Cp * hd

        self.Dt_l = Dt
        self.Xt_l = Xt
        # print("reward = ", reward)
        # print("error = ", hd)
        # print('robot = ',self.state[25], 'target = ',self.state[26])
        return reward, terminated, truncated


    def controll(self, action, from_action_dict=True):
        # print("action = ", action)
        action = np.array(action)
        if from_action_dict:
            move_vec = self.action_ves.get(np.argmax(action))
            # move_vec = action 
        else:
            move_vec = np.zeros((3))
            move_vec[0] = constrain(action[0], -self.max_vx/2, self.max_vx)
            move_vec[1] = constrain(action[1], -self.max_vy, self.max_vy)
            move_vec[2] = constrain(action[2], -self.max_w,  self.max_w)
        
        # print("move_vec = ", move_vec)
        self.joit_vec_ust = self.move_vec_to_joit_vec(move_vec)
        # print('joit_vec_ust =', self.joit_vec_ust)
        
    
    def integrate_ang_vel(self, joit_vec_ust):
        if self.render_fps > 0:
            err = joit_vec_ust - self.joit_ang_vel 
            filter = np.zeros((4))
            for i in range(len(filter)):
                if err[i] > 0.1:
                    filter[i] = 1
                elif err[i] < -0.1:
                    filter[i] = -1
                else:
                    filter[i] = 0.0

            # print('filter =', filter)
            wheel_eps_vec = np.array([self.wheel_eps, self.wheel_eps, self.wheel_eps, self.wheel_eps])
            self.joit_ang_vel += (filter * wheel_eps_vec/self.render_fps)
        else:
            self.joit_ang_vel = np.zeros((4))


    def move_vec_to_joit_vec(self, move_vec):
        move_vec_np = np.array(move_vec)
        l = self.length_wheel
        w = self.width_wheel
        H = np.array([[1, -1, -(l + w)],
                      [1,  1,  (l + w)],
                      [1,  1, -(l + w)],
                      [1, -1,  (l + w)]])
        joit_vec = 1/self.wheel_radius * (H @ move_vec_np.T)

        for i in range(len(joit_vec)):
            if joit_vec[i] > self.max_wheel_vel:
                joit_vec[i] = self.max_wheel_vel
            elif joit_vec[i] < -self.max_wheel_vel:
                joit_vec[i] = -self.max_wheel_vel


        return joit_vec
    
    
    def joit_vec_to_move_vec(self, joit_vec):
        joit_vec_np = np.array(joit_vec)
        l = self.length_wheel
        w = self.width_wheel
        
        self.Vx = self.wheel_radius/4 * np.sum(np.array([1, 1, 1, 1]) *  joit_vec_np)
        self.Vy = self.wheel_radius/4 * np.sum(np.array([-1, 1, 1, -1]) *  joit_vec_np)
        self.W = (self.wheel_radius/(4 * (l + w))) * np.sum(np.array([-1, 1, -1, 1]) *  joit_vec_np)
        
        move_vec = np.array([self.Vx, self.Vy, self.W])

        return move_vec

        

    def teleop(self, teleop_vec):
        scale = 4
        # print(self.transform)
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
        
            
        if self.render_fps > 0:
            move_vector = move_vector * METRIC_KF / self.render_fps
            W = teleop_vec[2] / self.render_fps
        else:
            move_vector = np.zeros(2)
            W = 0

        teleop_transform =  get_transform(move_vector , W)
        self.transform = self.transform @ teleop_transform
        # print(get_XYTheta(self.transform))
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        self.trajectory.append(np.array(self.get_pose())[0:2])

    def draw(self, screen, color = robot_color):
        # Drowing robot borders
        pg.draw.aaline(screen, color, (self.edge_points[0][0], self.edge_points[0][1]), (self.edge_points[1][0], self.edge_points[1][1]))
        pg.draw.aaline(screen, color, (self.edge_points[1][0], self.edge_points[1][1]), (self.edge_points[2][0], self.edge_points[2][1]))
        pg.draw.aaline(screen, color, (self.edge_points[2][0], self.edge_points[2][1]), (self.edge_points[3][0], self.edge_points[3][1]))
        pg.draw.aaline(screen, color, (self.edge_points[3][0], self.edge_points[3][1]), (self.edge_points[0][0], self.edge_points[0][1]))


        self.lidar.draw(screen)

        # draw trajectory
        for i in range(1,len(self.trajectory)):
            # self.trajectory[i]
            pg.draw.aaline(screen, robot_color, self.trajectory[i-1], self.trajectory[i])


        if self.collision[0]:
            pg.draw.circle(screen, way_color, [self.collision[1] , self.collision[2]], 7, 3)

        if self.DRAW_TARGET:
            target_arrow = [self.target[0] + 25 * np.cos(self.target[2]), self.target[1] + 25 * np.sin(self.target[2])]
            pg.draw.circle(screen, way_color, [self.target[0], self.target[1]], 7, 3)
            # pg.draw.aaline(screen, way_color, (self.target[0], self.target[1]), (target_arrow[0], target_arrow[1]))
        

        

    def get_pose(self):
        return self.x , self.y, self.theta
    
    def get_vel(self):
        return self.Vx , self.Vy, self.W
    
    def get_waypoint_in_local(self):
        way_point = np.array(self.way_point) * 10
        way_point[2] = 1
        way_point_loc = np.linalg.inv(self.transform) @ way_point.T
        way_point_loc = way_point_loc[:2]
        # print(way_point_loc)
        return way_point_loc.T
    

    

    

    






