import pygame as pg
from tkinter import *
import numpy as np
import sys
import os
import pandas as pd
import threading
from random import randint, random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pygame.locals import *
from utils.utils import *
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

class Action:
    def __init__(self):
        self.n = 6

    def sample(self):
        return randint(0,5)

class Simulator:
    def __init__(self):
        
        self.FPS = 30

        self.WINDOW_SIZE = (1800, 1000)
        self.target = [self.WINDOW_SIZE[0]//2, self.WINDOW_SIZE[1]//2, 0.0]

        # init objects
        # self.map = Map(self.WINDOW_SIZE)
        # self.robot = Robot(self.map, init_pos=[100,100, 1])
        # self.robot.target = self.target



        # pg.init()
        # self.screen = pg.display.set_mode(self.WINDOW_SIZE,RESIZABLE, 32)
        # self.clock = pg.time.Clock()
        # self.font = pg.font.SysFont("Ubuntu Condensed", 14, bold=False, italic=False)

        # Learning parametrs
        self.observation_space = np.zeros((16))
        self.observation = []

        self.action_space = Action()
        self.action = self.action_space.sample()

        self.reward_range = (-np.inf, np.inf)

        self.old_robots = []
        self.reset()
        

    def init_window(self):
        pg.init()
        self.screen = pg.display.set_mode(self.WINDOW_SIZE,RESIZABLE, 32)
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Ubuntu Condensed", 14, bold=False, italic=False)


    def kill_window(self):
        pg.quit()
        # sys.exit()


    
    def reset(self):
        # init objects
        self.map = Map(self.WINDOW_SIZE)
        # init_pos = [randint(100,1700), randint(100,900), (random()-0.5)*2*np.pi]
        init_pos = [1600, 800, (random()-0.5)*2*np.pi]
        self.robot = Robot(self.map, init_pos=init_pos)
        self.robot.set_target(self.target)
        self.old_robots.append(self.robot)
        # self.pygame_iter() # Обновляем состояние среды
        state = self.robot.get_state()
        info = {}
        return state, info

    def step(self, action):
        self.robot.controll(action) # Перемещаем робота на одно действие
        self.pygame_iter() # Обновляем состояние среды
        next_state = self.robot.get_state()
        # print(next_state)
        reward, terminated, truncated = self.robot.get_reward()
        info = {}
        return next_state, reward, terminated, truncated, info

    def iteration(self):
        # thread_pygame = threading.Thread(target=self.pygame_window, args=())
        # thread_tkinter = threading.Thread(target=self.tkinter_window, args=())
        # thread_pygame.start()
        # # thread_tkinter.start()
        self.pygame_iter()

    def get_robot_data(self):
        return self.robot.get_pose()


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

        
        if True:
            # self.clock.tick(self.FPS)
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
                self.robot.teleop(teleop_vec=[1,0,0])
            if keys[pg.K_s]:
                self.robot.teleop(teleop_vec=[-1,0,0])
            if keys[pg.K_a]:
                self.robot.teleop(teleop_vec=[0,-1,0])
            if keys[pg.K_d]:
                self.robot.teleop(teleop_vec=[0,1,0])
            if keys[pg.K_q]:
                self.robot.teleop(teleop_vec=[0,0,-0.15])
            if keys[pg.K_e]:
                self.robot.teleop(teleop_vec=[0,0,0.15])



        # Render scene
            self.map.update()
            self.robot.update(self.map)

        # Update display
            self.screen.fill(silver)

            self.robot.draw(self.screen)
            self.map.draw(self.screen)
            if len(self.old_robots) > 15:
                for i in range(len(self.old_robots)-15,len(self.old_robots)):
                    self.old_robots[i].draw(self.screen)

            x,y,theta = self.robot.get_pose()
            text = self.font.render(f'Robot coordinates: x = {x:.2f}, y = {y:.2f}, theta = {theta:.2f}' , True, black)
            self.screen.blit(text, [10,10])
            text = self.font.render(f'Steps: x = {self.robot.n_steps}' , True, black)
            self.screen.blit(text, [10,25])

            pg.display.update()


