import pygame as pg
from tkinter import *
import numpy as np
import sys
import os
import pandas as pd
import threading
import random
import time

SIM_TIME = True
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pygame.locals import *
from utils.transforms import *
from robot.robot import *
from environment.map import *
from environment.obstacles import *

def create_folder(workspace:str, folder:str) -> None:
    path = os.path.join(workspace, folder)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"create folder with path {0}".format(path))
    return path

def distance(point_1, point_2):
    return np.sqrt(np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]))

class Action:
    def __init__(self):
        self.n = 6

    def sample(self):
        return random.randint(0,5)

class Simulator:
    def __init__(self, render_fps=30, **kwargs):
        self.global_map = None
        self.FPS = render_fps
        seed = kwargs["seed"]
        if seed is not None:
            random.seed(seed)
        self.WINDOW_SIZE = (1600, 400)
        self.target = [self.WINDOW_SIZE[0]//2, self.WINDOW_SIZE[1]//2, 0.0]
        if 'robot_init_pos' in kwargs:
            self.robot_init_pos = kwargs['robot_init_pos']
        else:
            self.robot_init_pos = None

        if 'robot_goal_pos' in kwargs:
            self.robot_goal_pos = kwargs['robot_goal_pos']
        else:
            self.robot_goal_pos = None

        # Learning parametrs
        self.observation_space = np.zeros((16))
        self.observation = []

        self.action_space = Action()
        self.action = self.action_space.sample()

        self.reward_range = (-np.inf, np.inf)

        self.n_robots = 10
        self.robots = []

        self.old_robots = []
        self.reset()
        # self.t_old = time.time()
        # fps = time.time() - self.t_old
        self.init_window()
        

    def init_window(self):
        pg.init()
        self.screen = pg.display.set_mode(self.WINDOW_SIZE,RESIZABLE, 32)
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Ubuntu Condensed", 20, bold=False, italic=False)


    def kill_window(self):
        pg.quit()
        # sys.exit()


    
    def reset(self):
        # init objects
        self.map = Map(size=self.WINDOW_SIZE)
        self.map.set_global_map(self.global_map)
        moveable_obstacles = self.map.get_moveable_obstacles()
        if self.robot_init_pos is None:
            while True:
                x = random.randint(0, self.WINDOW_SIZE[0]-1)
                y = random.randint(0, self.WINDOW_SIZE[1]-1)
                distances_robot_obstacle = []
                for obstacle in moveable_obstacles:
                    distances_robot_obstacle.append(distance([x,y], [obstacle.x, obstacle.y]))
                if not self.map.bin_map_og[y,x] and min(distances_robot_obstacle) > 100:
                    break
            fi = (random.random()-0.5)*2*np.pi
            init_pos = [x, y, fi]
            self.robot = Robot(self.map, init_pos=init_pos)
        else:
            self.robot = Robot(self.map, init_pos=self.robot_init_pos)
            init_pos = self.robot_init_pos

        if self.robot_goal_pos is None:
            while True:
                x = random.randint(0, self.WINDOW_SIZE[0]-1)
                y = random.randint(0, self.WINDOW_SIZE[1]-1)
                distance_robot_target = distance([init_pos[0],init_pos[1]], [x,y])
                if not self.map.bin_map_og[y,x] and distance_robot_target > 100 and distance_robot_target < 300:
                    break
            fi = (random.random()-0.5)*2*np.pi
            self.robot.set_target([x, y, fi])
        else:
            self.robot.set_target(self.robot_goal_pos)

        # self.old_robots.append(self.robot)
        # self.pygame_iter() # Обновляем состояние среды
        state, info = self.robot.get_state()

        return state, info

    def step(self, action, pid_mode=False):
        self.robot.controll(action, from_action_dict=False, pid_mode=pid_mode) # Перемещаем робота на одно действие

        self.pygame_iter() # Обновляем состояние среды
        next_state, info_s = self.robot.get_state()
        # print(next_state)
        reward, terminated, truncated, info_r = self.robot.get_reward()
        info_s.update(info_r)
        return next_state, reward, terminated, truncated, info_s

    def iteration(self):
        # thread_pygame = threading.Thread(target=self.pygame_window, args=())
        # thread_tkinter = threading.Thread(target=self.tkinter_window, args=())
        # thread_pygame.start()
        # # thread_tkinter.start()
        self.pygame_iter()

    def get_robot_data(self):
        return self.robot.get_pose()
    
    def set_global_map(self, global_map):
        self.global_map = global_map

    def tkinter_window(self):
        # create root window
        root = Tk()

        # root window title and dimension
        root.title("Welcome to GeekForGeeks")
        # Set geometry (widthxheight)
        root.geometry('350x200')

        # adding menu bar in root window
        # new item in menu bar labelled as 'New'
        # adding more items in the menu bar 
        menu = Menu(root)
        item = Menu(menu)
        item.add_command(label='New')
        menu.add_cascade(label='File', menu=item)
        root.config(menu=menu)

        # adding a label to the root window
        lbl = Label(root, text = "Are you a Geek?")
        lbl.grid()

        # adding Entry Field
        txt = Entry(root, width=10)
        txt.grid(column =1, row =0)


        # function to display user text when
        # button is clicked
        def clicked():

            res = "You wrote" + txt.get()
            lbl.configure(text = res)

        # button widget with red color text inside
        btn = Button(root, text = "Click me" ,
                    fg = "red", command=clicked)
        # Set Button Grid
        btn.grid(column=2, row=0)

        # all widgets will be here
        # Execute Tkinter
        root.mainloop()

    def pygame_iter(self):

        # Check events
            for i in pg.event.get():
                if i.type == QUIT:
                    pg.quit()
                    sys.exit()
                elif i.type == KEYDOWN:
                    # print(i.key)
                    if i.key == 27:
                        pg.quit()
                        sys.exit()
                    if i.key == 32:
                        pass
                    if i.key == 13:
                        self.robot.auto_mode = not self.robot.auto_mode

            keys = pg.key.get_pressed()
            if keys[pg.K_w]:
                self.robot.controll([1, 0, 0], False) # In m/s, m/s, rad/s
            if keys[pg.K_s]:
                self.robot.controll([-1, 0, 0], False)
            if keys[pg.K_a]:
                self.robot.controll([0, -1, 0], False)
            if keys[pg.K_d]:
                self.robot.controll([0, 1, 0], False)
            if keys[pg.K_q]:
                self.robot.controll([0, 0, -1], False)
            if keys[pg.K_e]:
                self.robot.controll([0, 0, 1], False)


            self.map.update(render_fps=self.clock.get_fps())
            self.robot.update(map=self.map, render_fps=self.clock.get_fps())
            # for robot in self.robots:
            #     robot.update(self.map)

        # Render scene
    def render(self):

            if SIM_TIME:
                self.clock.tick(self.FPS)

            self.screen.fill(silver)

            self.map.draw(self.screen)

        # Update display


            self.robot.draw(self.screen)
            # for robot in self.robots:
            #     robot.draw(self.screen)

            # if len(self.old_robots) > 15:
            #     for i in range(len(self.old_robots)-15,len(self.old_robots)):
            #         self.old_robots[i].draw(self.screen)

            x,y,theta = self.robot.get_pose()
            Vx, Vy, W = self.robot.get_vel()
            text = self.font.render(f'Robot coordinates: x = {x:.2f}, y = {y:.2f}, theta = {theta:.2f}' , True, black)
            self.screen.blit(text, [30,10])
            text = self.font.render(f'Robot coordinates: Vx = {Vx:.2f}, Vy = {Vy:.2f}, W = {W:.2f}' , True, black)
            self.screen.blit(text, [30,35])
            text = self.font.render(f'Steps: {self.robot.n_steps}, Sec: {self.robot.live_secs:.2f} ' , True, black)
            self.screen.blit(text, [30,65])
            text = self.font.render(f'FPS: {self.clock.get_fps():.2f}' , True, black)
            self.screen.blit(text, [30,95])
            text = self.font.render(f'Instantaneous reward: {self.robot.instantaneous_reward:.16f}' , True, black)
            self.screen.blit(text, [30,125])


            # print(self.clock.get_fps())

            pg.display.update()


