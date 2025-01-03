import pygame as pg
from tkinter import *
import numpy as np
import sys
import os
import pandas as pd
import threading

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

class Simulator:
    def __init__(self):
        
        self.FPS = 30

        self.WINDOW_SIZE = (1800, 1000)

        # init objects
        self.map = Map(self.WINDOW_SIZE)
        self.map.generate()

        self.robots = []
        NUM_ROBOTS = 1
        for i in range(NUM_ROBOTS):
            robot = Robot(self.map.bool_map)
            self.robots.append(robot)

        # robot = Robot(map.bool_map)
        #Cтены
        obstacle_l = Obstacle(init_pos=[30//2, 1000//2, 0.001], init_size=[30, 1000])
        obstacle_r = Obstacle(init_pos=[1800-30//2, 1000//2, 0.001], init_size=[30, 1000])
        obstacle_t = Obstacle(init_pos=[1800//2, 30//2, 0.001], init_size=[1800, 30])
        obstacle_b = Obstacle(init_pos=[1800//2, 1000-30//2, 0.001], init_size=[1800, 30])

        obstacle_0 = Obstacle(init_pos=[200, 330//2+30, 0.001], init_size=[30, 330])
        obstacle_1 = Obstacle(init_pos=[200+800//2, 330+30//2, 0.001], init_size=[800, 30])
        obstacle_2 = Obstacle(init_pos=[400//2, 530+30//2, 0.001], init_size=[400, 30])
        obstacle_3 = Obstacle(init_pos=[400+150+500//2, 530+30//2, 0.001], init_size=[500, 30])
        obstacle_4 = Obstacle(init_pos=[400+150+700, 750+30//2, -2.3], init_size=[600, 30])
        obstacle_5 = Obstacle(init_pos=[1000-30//2, 230//2+30, 0.001], init_size=[30, 230])
        # obstacle_6 = Obstacle(init_pos=[100, 500, 5])
        # obstacle_7 = Obstacle(init_pos=[500, 500, 0.001])
        # obstacle_8 = Obstacle(init_pos=[200, 200, 0.001])
        # obstacle_9 = Obstacle(init_pos=[500, 100, 1.571])
        # obstacle_10 = Obstacle(init_pos=[100, 500, 5])
        # obstacle_11 = Obstacle(init_pos=[500, 500, 0.001])

        self.map.add_obstacle(obstacle_l)
        self.map.add_obstacle(obstacle_r)
        self.map.add_obstacle(obstacle_t)
        self.map.add_obstacle(obstacle_b)

        self.map.add_obstacle(obstacle_0)
        self.map.add_obstacle(obstacle_1)
        self.map.add_obstacle(obstacle_2)
        self.map.add_obstacle(obstacle_3)
        self.map.add_obstacle(obstacle_4)
        self.map.add_obstacle(obstacle_5)
        # map.add_obstacle(obstacle_6)
        # map.add_obstacle(obstacle_7)
        # map.add_obstacle(obstacle_8)
        # map.add_obstacle(obstacle_9)
        # map.add_obstacle(obstacle_10)
        # map.add_obstacle(obstacle_11)

        pg.init()
        self.screen = pg.display.set_mode(self.WINDOW_SIZE,RESIZABLE, 32)
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Ubuntu Condensed", 14, bold=False, italic=False)


    def iteration(self):
        # thread_pygame = threading.Thread(target=self.pygame_window, args=())
        # thread_tkinter = threading.Thread(target=self.tkinter_window, args=())


        # thread_pygame.start()
        # # thread_tkinter.start()
        # self.pygame_window()
        self.pygame_window()

    def get_robot_data(self):
        return self.robots[0].get_pose()


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

    def pygame_window(self):

        
        if True:
            self.clock.tick(self.FPS)
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
                        robot[0].auto_mode = not robot[0].auto_mode

            keys = pg.key.get_pressed()
            if keys[pg.K_w]:
                self.robots[0].teleop(teleop_vec=[1,0,0])
            if keys[pg.K_s]:
                self.robots[0].teleop(teleop_vec=[-1,0,0])
            if keys[pg.K_a]:
                self.robots[0].teleop(teleop_vec=[0,-1,0])
            if keys[pg.K_d]:
                self.robots[0].teleop(teleop_vec=[0,1,0])
            if keys[pg.K_q]:
                self.robots[0].teleop(teleop_vec=[0,0,-0.15])
            if keys[pg.K_e]:
                self.robots[0].teleop(teleop_vec=[0,0,0.15])



        # Render scene
            self.map.update()
            for robot in self.robots:
                robot.update(self.map)

            #     x = int(robot.get_pose()[0])
            #     y = int(robot.get_pose()[1])
            #     print(np.max(map.map_d))
            #     if map.map_d[x][y] == 0:
            #         robot.robot_points.append([robot.get_pose()[:2], 0])
            #     else:
            #         robot.robot_points.append([robot.get_pose()[:2], 1])

        # Update display
            self.screen.fill(silver)

            # map.draw(screen)
            for robot in self.robots:
                robot.draw(self.screen)
            # robot.draw(screen)
            self.map.draw(self.screen)


            x,y,theta = self.robots[0].get_pose()
            text = self.font.render(f'Robot coordinates: x = {x:.2f}, y = {y:.2f}, theta = {theta:.2f}' , True, black)
            self.screen.blit(text, [10,10])
            text = self.font.render(f'Robot coordinates on map: x = {self.robots[0].robot_pose_on_map[0]}, y = {self.robots[0].robot_pose_on_map[1]}' , True, black)
            self.screen.blit(text, [10,25])

            pg.display.update()


sim = Simulator()
while True:
    sim.iteration()
    print(sim.get_robot_data())