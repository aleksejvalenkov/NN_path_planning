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


class DepthSensor:
    def __init__(self, transform) -> None:
        self.fov = 150 # in degree
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
                    if bool_map[x][y] == 1:
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
        # for ray in self.rays_angles:
        #     pg.draw.aaline(screen, robot_color, [self.x, self.y], [self.x + np.cos(ray+self.theta) * self.ray_lenght , self.y + np.sin(ray+self.theta) * self.ray_lenght])
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
        self.throttle = 2
        self.bool_map = bool_map
        self.x , self.y, self.theta = [100,100, 0]
        self.t_vec = np.array([ self.x , self.y])
        self.transform = get_transform(self.t_vec, self.theta)
        self.path = None
        self.way_point = (13, 13, 2)
        # print(f'Robot transform: \n {self.transform}')
        #Init Sensors
        self.camera_transform = np.array( 
            [[  1., 0., self.robot_radius ],
             [  0., 1., 0. ],
             [  0., 0., 1. ]]
             )
        self.depth_camera = DepthSensor(self.transform @ self.camera_transform)
        self.depth_image = None

        self.pid_theta = PID(0.5, 0.0001, 0, 0)
        self.pid_theta.sample_time = 0.033

    def update(self, map):
        self.cell_size = round(1/map.scale)

        self.depth_camera.transform = self.transform @ self.camera_transform
        self.depth_camera.update()
        self.depth_image = self.depth_camera.scan(self.bool_map)
        # print(f'Robot pose on map {(self.x//10, self.y//10)}')
        code = self.code_from_theta(self.theta)
        self.robot_pose_on_map = (int(self.x//10), int(self.y//10), code)
        if map.resized_map[self.way_point[0]][self.way_point[1]] != 1:
            self.path = solve(map.resized_map, self.robot_pose_on_map, self.way_point)

        action, text = self.get_action_from_path(self.path, self.robot_pose_on_map)
        print(f'Action: {action}, {text}')

        self.controll(action)

    def controll(self, action):
        d_theta = 0.0
        self.throttle = 0.0
        code = self.code_from_theta(self.theta)
        self.robot_pose_on_map = (int(self.x//10), int(self.y//10), code)
        d_theta = self.get_global_angle(self.path, self.robot_pose_on_map)
        if action == 1:
            self.throttle = 1
        elif action == 3:
            self.throttle = 1
        elif action == 2:
            self.throttle = 1
        elif action == 0:
            self.throttle = 0.0
        elif action == 4:
            self.throttle = 0.0
        elif action == 5:
            self.throttle = 0.0

        # print(d_theta)
        self.pid_theta.setpoint = d_theta

        if self.theta > math.pi - math.pi / 4 and d_theta < 0:
            theta = (self.theta * -1) - math.pi / 2
        elif self.theta < -math.pi + math.pi / 4 and d_theta > 0:
            theta = (self.theta * -1) + math.pi / 2
        else:
            theta = self.theta
        
        # print('self.theta ', self.theta)
        # print('     theta ', theta)
        self.theta_speed = constrain(self.pid_theta(theta), -0.1, 0.1)
        if action != 5:
            self.teleop(teleop_vec=[self.throttle,0,self.theta_speed])

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

        c = self.cell_size
        pg.draw.rect(screen, robot_color, (self.robot_pose_on_map[0]*c, self.robot_pose_on_map[1]*c, c, c))
        # Draw path
        if self.path is not None:
            for cell in self.path:
                pg.draw.rect(screen, way_color, (cell[0]*c, cell[1]*c , c, c))

    def get_pose(self):
        return self.x , self.y, self.theta
    
    def code_from_theta(self, theta):
        code = 0
        theta = np.degrees(theta) * -1
        if -22.5 < theta <= 22.5:
            code = 0
        if 22.5 < theta <= 67.5:
            code = 1
        if 67.5 < theta <= 112.5:
            code = 2
        if 112.5 < theta <= 157.5:
            code = 3
        if 157.5 < theta or theta < -157.5:
            code = 4
        if -22.5 > theta >= -67.5:
            code = 7
        if -67.5 > theta >= -112.5:
            code = 6
        if -112.5 > theta >= -157.5:
            code = 5
        return code
    
    def theta_from_code(self, code):
        theta = 0.0
        if code == 0:
            theta = 0.0
        elif code == 1:
            theta = -math.pi / 4
        elif code == 2:
            theta = -math.pi / 2
        elif code == 3:
            theta = -math.pi / 2 - math.pi / 4
        elif code == 4:
            if self.theta > 0:
                theta = math.pi 
            else:
                theta = -math.pi 
        elif code == 5:
            theta = math.pi / 2 + math.pi / 4
        elif code == 6:
            theta = math.pi / 2
        elif code == 7:
            theta = math.pi / 4
        

        return theta
    
    def get_global_angle(self, path, robot_pose):
        angle = 0.0
        if path:
            first_step = path[0]
            d = np.array(first_step) - np.array(robot_pose)
            new_code = robot_pose[2] + d[2]
            if d[2] == 7 or d[2] == -7:
                d[2] = -1
            new_code = robot_pose[2] + d[2]
            if new_code == 8:
                new_code = 1
            if new_code == -1:
                new_code = 7
            print(d)
            if d[0] == 0 and  d[1] == 1:
                angle = math.pi / 2
            elif d[0] == 1 and  d[1] == 1:
                angle = math.pi / 4
            elif d[0] == 1 and  d[1] == 0:
                angle = 0.0
            elif d[0] == 1 and  d[1] == -1:
                angle = -math.pi / 4
            elif d[0] == 0 and  d[1] == -1:
                angle = -math.pi / 2
            elif d[0] == -1 and  d[1] == -1:
                angle = -math.pi / 2 -math.pi / 4
            elif d[0] == -1 and  d[1] == 0:
                if self.theta > 0:
                    angle = math.pi
                else:
                    angle = -math.pi
            elif d[0] == -1 and  d[1] == 1:
                angle = math.pi / 4 + math.pi / 2
            elif d[0] == 0 and  d[1] == 0 and d[2] == 1:
                angle = self.theta_from_code(new_code)
            elif d[0] == 0 and  d[1] == 0 and d[2] == -1:
                angle = self.theta_from_code(new_code)

        return angle

    def get_action_from_path(self, path, robot_pose):
        action, text = None, None
        if path:
            first_step = path[0]
            d = np.array(first_step) - np.array(robot_pose)
            if d[2] == 7 or d[2] == -7:
                d[2] = -1
            # print(np.abs(d[:2]))

            if d[2] == 0:
                action = 2
                text = 'forward'
            elif np.sum(np.abs(d[:2])) == 0:
                if d[2] == 1:
                    action = 0
                    text = 'stay and rot -45*'
                if d[2] == -1:
                    action = 4
                    text = 'stay and rot 45*'
            elif np.sum(np.abs(d[:2])) > 0 and d[2] != 0:
                if d[2] == 1:
                    action = 1
                    text = 'forward and rot -45*'
                if d[2] == -1:
                    action = 3
                    text = 'forward and rot 45*'
        else:
            action = 5
            text = 'stay'

        return action, text


class Map:
    def __init__(self, size = (200, 200)) -> None:
        self.size = size
        self.map = np.zeros(self.size)
        self.bool_map = np.zeros(self.size)
        self.scale = 0.1
        self.expansion_cells = 3
        self.resized_map = cv2.resize(self.bool_map, (int(self.size[0]*self.scale) , int(self.size[1]*self.scale)), interpolation = cv2.INTER_NEAREST)
        pass

    def generate(self):
        def get_bool_map(map):
            bool_map = [ [False for j in range(len(map[0]))] for i in range(len(map))]
            for i in range(len(map)):
                for j in range(len(map[0])):
                    if map[i][j] >= 0.5:
                        bool_map[i][j] = 1
                    else:
                        bool_map[i][j] = 0
            return bool_map
        
        def get_image_map(map_, resized_map):
            cell_size = round(1/self.scale)
            map = copy.copy(map_)
            for i in range(0, len(map), cell_size):
                for j in range(0, len(map[0]), cell_size):
                    for k in range(i, i+cell_size):
                        for l in range(j, j+cell_size):
                            if resized_map[k//cell_size][l//cell_size] == 1 and map[k][l]!=1:
                                map[k][l] = 2

            img_1 = np.zeros([self.size[0], self.size[1], 3])
            for i in range(len(map)):
                for j in range(len(map[0])):
                    if map[i][j] == 0:
                        img_1[i][j] = silver
                    elif map[i][j] == 2:
                        img_1[i][j] = map_color_2
                    elif map[i][j] == 1:
                        img_1[i][j] = map_color
            return img_1
        
        def extend_bool_map(bool_map):
            def get_cels_in_radius(v, map):
                x1, y1 = v
                rows, cols = map.shape
                check_next_node = lambda x, y: True if 0 <= y < cols and 0 <= x < rows and not bool(map[x][y]) else False
                ways = [-1, 0], [0, -1], [1, 0], [0, 1], [-1, -1], [1, -1], [1, 1], [-1, 1]
                return [(x1 + dx, y1 + dy) for dx, dy in ways if check_next_node(x1 + dx, y1 + dy)]
                
            extended_bool_map = copy.copy(bool_map)

            for i in range(self.expansion_cells):
                if i + 1 == 1:
                    for (x, y), value in np.ndenumerate(bool_map):
                        if bool_map[x][y] == 1:
                            for cell in get_cels_in_radius((x, y), bool_map):
                                # print(cell)
                                extended_bool_map[cell[0]][cell[1]] = 1
                else:
                    Map_F = copy.copy(bool_map)
                    for (x, y), value in np.ndenumerate(extended_bool_map):
                        if extended_bool_map[x][y] == 1:
                            for cell in get_cels_in_radius((x, y), extended_bool_map):
                                # print(cell)
                                Map_F[cell[0]][cell[1]] = 1
                    for (x, y), value in np.ndenumerate(bool_map):
                        extended_bool_map[x][y] = extended_bool_map[x][y] or Map_F[x][y]

            return extended_bool_map

        np.random.seed(0)
        noise = generate_perlin_noise_2d(self.size, (self.size[0] // 100, self.size[1] // 100))
        noise[noise < 0.5 ] = 0
        noise[noise >= 0.5 ] = 1
        # print(noise)
        self.bool_map = get_bool_map(noise)
        self.bool_map = noise
        self.resized_map = cv2.resize(self.bool_map, (int(self.size[0]*self.scale) , int(self.size[1]*self.scale)), interpolation = cv2.INTER_NEAREST)
        self.resized_map = extend_bool_map(self.resized_map)
        self.map = get_image_map(self.bool_map, self.resized_map)
   

    def update(self):
        # self.path = solve(self.resized_map, (0 , 0), (10 , 10))
        pass

    def draw(self, screen):
        surf = pg.surfarray.make_surface(self.map)
        screen.blit(surf, (0, 0))


