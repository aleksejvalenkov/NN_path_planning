import pygame as pg
import numpy as np
from numpy import floor
from transforms import *
from perlin_numpy import generate_perlin_noise_2d
import cv2 
import copy

map_color = (100,100,100)
robot_color = (0,0,190)
sensor_color = (0,190,0)
silver = (194, 194, 194)


class DepthSensor:
    def __init__(self, transform) -> None:
        self.fov = 100 # in degree
        self.fov_rad = np.radians(self.fov) # in radians
        self.pix = 40 # pixels in 1d image from camera
        self.transform = transform
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        self.rays_angles = np.arange(start=-self.fov_rad/2, stop=self.fov_rad/2, step=self.fov_rad/self.pix)
        self.ray_lenght = 500
        self.matrix = np.zeros(self.pix)
        self.range_points = np.zeros((self.pix, 2))

    def update(self):
        self.x , self.y, self.theta = get_XYTheta(self.transform)
        # print(self.matrix)

    def scan(self, bool_map):
        for i_ray in range(len(self.rays_angles)):
            loc_points = []
            k = np.tan(self.rays_angles[i_ray])
            for x in range(0, self.ray_lenght, 2):
                y = k*x
                loc_points.append(np.array([x,y,1]).T)
            
            loc_points = np.array(loc_points)

            for loc_point in loc_points:
                point = self.transform @ loc_point
                x, y, _ = point
                x = round(x)
                y = round(y)
                x_size = len(bool_map) - 1
                y_size = len(bool_map[0]) - 1
                if 0 <= x <= x_size and 0 <= y <= y_size:
                    if bool_map[x][y] == False:
                        self.range_points[i_ray] = (x,y)
                        break
                    else:
                        self.range_points[i_ray] = (np.inf, np.inf)

            self.matrix[i_ray] = np.sqrt(pow((self.x - self.range_points[i_ray][0]), 2) + pow((self.y - self.range_points[i_ray][1]), 2))
        return self.matrix

    def draw(self, screen):
        # draw sensor
        pg.draw.circle(screen, sensor_color, [self.x , self.y], 7, 3)
        # draw rays
        for ray in self.rays_angles:
            pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(ray+self.theta) * self.ray_lenght , self.y + np.sin(ray+self.theta) * self.ray_lenght])
        for point in self.range_points:
            pg.draw.circle(screen, (255,0,0), [point[0] , point[1]], 3, 3)

        # draw SensorBar
        screen_size = screen.get_size()
        rect_size = 20
        lens = copy.copy(self.matrix)
        lens[lens >= self.ray_lenght] = self.ray_lenght
        lens[:] = lens[:] * 255 / 500
        # print(lens)
        for i in range(len(lens)):
            r = round(lens[i])
            pg.draw.rect(screen, (r, r, r), 
                 (rect_size*i, screen_size[1]-rect_size , rect_size, rect_size))

class UWBSensor:
    def __init__(self) -> None:
        pass
    

class Robot:
    def __init__(self, bool_map) -> None:
        self.robot_radius = 25
        self.bool_map = bool_map
        self.x , self.y, self.theta = [100,100, 2]
        self.t_vec = np.array([ self.x , self.y])
        self.transform = get_transform(self.t_vec, self.theta)
        # print(f'Robot transform: \n {self.transform}')
        #Init Sensors
        self.camera_transform = np.array( 
            [[  1., 0., self.robot_radius ],
             [  0., 1., 0. ],
             [  0., 0., 1. ]]
             )
        self.depth_camera = DepthSensor(self.transform @ self.camera_transform)
        self.depth_image = None

    def update(self):
        self.depth_camera.transform = self.transform @ self.camera_transform
        self.depth_camera.update()
        self.depth_image = self.depth_camera.scan(self.bool_map)

    def teleop(self, teleop_vec):
        self.t_vec = np.array([ teleop_vec[0] , teleop_vec[1]])
        teleop_transform =  get_transform(self.t_vec, teleop_vec[2])
        self.transform = self.transform @ teleop_transform
        # print(get_XYTheta(self.transform))
        self.x , self.y, self.theta = get_XYTheta(self.transform)

    def draw(self, screen):

        pg.draw.circle(screen, robot_color, (self.x , self.y), self.robot_radius, 5)
        pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(self.theta) * self.robot_radius , self.y + np.sin(self.theta) * self.robot_radius])

        self.depth_camera.draw(screen)

    def get_pose(self):
        return self.x , self.y, self.theta

class Map:
    def __init__(self, size = (200, 200)) -> None:
        self.size = size
        self.map = np.zeros(self.size)
        pass

    def generate(self):
        def get_bool_map(map):
            bool_map = [ [False for j in range(len(map[0]))] for i in range(len(map))]
            for i in range(len(map)):
                for j in range(len(map[0])):
                    if map[i][j] >= 0.5:
                        bool_map[i][j] = False
                    else:
                        bool_map[i][j] = True

            return bool_map
        
        def get_image_map(map):
            img_1 = np.zeros([self.size[0], self.size[1], 3])
            for i in range(len(map)):
                for j in range(len(map[0])):
                    if map[i][j] >= 0.5:
                        img_1[i][j] = map_color
                    else:
                        img_1[i][j] = silver

            return img_1

        # np.random.seed(0)
        noise = generate_perlin_noise_2d(self.size, (self.size[0] // 100, self.size[1] // 100))
        noise[noise < 0.5 ] = 0
        noise[noise >= 0.5 ] = 1

        self.bool_map = get_bool_map(noise)
        self.map = get_image_map(noise)
   

    def update(self):
        pass

    def draw(self, screen):
        surf = pg.surfarray.make_surface(self.map)
        screen.blit(surf, (0, 0))