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

def get_max_index(path):
    files = os.listdir(path)
    print(len(files))


    return len(files)

FPS = 30
silver = (194, 194, 194)
black = (0, 0, 0)
red =   (194,0 , 0)
green = (0, 194, 0)
blue =  (0, 0, 194)

WINDOW_SIZE = (1800, 1000)
DATAPATH = '/home/alex/Documents/datasets/VisualPlanerData/iter_0'

index = get_max_index(DATAPATH) + 1
try:
    df = pd.read_csv(DATAPATH + '/' + 'out.csv')
except:
    df = pd.DataFrame(columns=['File Name', 'Label'])



def tkinter_window():
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

def pygame_window():
    pg.init()
    screen = pg.display.set_mode(WINDOW_SIZE,RESIZABLE, 32)
    clock = pg.time.Clock()
    font = pg.font.SysFont("Ubuntu Condensed", 14, bold=False, italic=False)


    # init objects
    map = Map(WINDOW_SIZE)
    map.generate()
    robot = Robot(map.bool_map)
    #Cтены
    obstacle_wall = Obstacle(init_pos=[1800//2,1000//2, 0], init_size=[1800,1000])
    # obstacle_r = Obstacle(init_pos=[200,200, 0])
    # obstacle_t = Obstacle(init_pos=[200,200, 0])
    # obstacle_b = Obstacle(init_pos=[200,200, 0])

    obstacle_0 = Obstacle(init_pos=[200,200, 0])
    obstacle_1 = Obstacle(init_pos=[500,100, 1.57])
    obstacle_2 = Obstacle(init_pos=[100,500, 5])
    obstacle_3 = Obstacle(init_pos=[500,500, 0])
    map.add_obstacle(obstacle_wall)
    map.add_obstacle(obstacle_0)
    map.add_obstacle(obstacle_1)
    map.add_obstacle(obstacle_2)
    map.add_obstacle(obstacle_3)
    record = False

    
    while True:
        clock.tick(FPS)
    # Check events
        for i in pg.event.get():
            if i.type == QUIT:
                csv = df.to_csv(DATAPATH + '/' + 'out.csv', index=False)
                pg.quit()
                sys.exit()
            elif i.type == KEYDOWN:
                # print(i.key)
                if i.key == 27:
                    csv = df.to_csv(DATAPATH + '/' + 'out.csv', index=False)
                    pg.quit()
                    sys.exit()
                if i.key == 32:
                    record = not record
                    if record:
                        print(f'Start recording')
                    else:
                        print(f'Stop recording')
                        csv = df.to_csv(DATAPATH + '/' + 'out.csv', index=False)
                        lpk = []
                        for p, c in robot.robot_points:
                            lpk.append(c)
                        F = 1 - (sum(lpk) / len(lpk))
                        print(f'F = {F}')
                if i.key == 13:
                    robot.auto_mode = not robot.auto_mode

        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            robot.teleop(teleop_vec=[1,0,0])
        if keys[pg.K_s]:
            robot.teleop(teleop_vec=[-1,0,0])
        if keys[pg.K_a]:
            robot.teleop(teleop_vec=[0,-1,0])
        if keys[pg.K_d]:
            robot.teleop(teleop_vec=[0,1,0])
        if keys[pg.K_q]:
            robot.teleop(teleop_vec=[0,0,-0.15])
        if keys[pg.K_e]:
            robot.teleop(teleop_vec=[0,0,0.15])



    # Render scene
        map.update()
        robot.update(map)
        

        if record:
            # print(robot.depth_image)
            if robot.action is not 6:
                obs = list(robot.get_waypoint_in_local())
                obs.extend(list(robot.depth_image))
                # obs.append(robot.action)
                data = np.array(obs)
                # print(data)
                print(f'Saved: {index}')
                # df.loc[len(df)] = [f'{index}.npy', robot.action]
                # np.save(DATAPATH + f'/{index}.npy', data)
                index += 1

            x = int(robot.get_pose()[0])
            y = int(robot.get_pose()[1])
            print(np.max(map.map_d))
            if map.map_d[x][y] == 0:
                robot.robot_points.append([robot.get_pose()[:2], 0])
            else:
                robot.robot_points.append([robot.get_pose()[:2], 1])

    # Update display
        screen.fill(silver)

        # map.draw(screen)
        robot.draw(screen)
        map.draw(screen)

        for p, c in robot.robot_points:
            if c == 0:
                pg.draw.circle(screen, blue, (p[0] , p[1]), 2)
            else:
                pg.draw.circle(screen, red, (p[0] , p[1]), 2)

        x,y,theta = robot.get_pose()
        text = font.render(f'Robot coordinates: x = {x:.2f}, y = {y:.2f}, theta = {theta:.2f}' , True, black)
        screen.blit(text, [10,10])
        text = font.render(f'Robot coordinates on map: x = {robot.robot_pose_on_map[0]}, y = {robot.robot_pose_on_map[1]}' , True, black)
        screen.blit(text, [10,25])
        if record:
            pg.draw.circle(screen, red, (WINDOW_SIZE[0]-20, 20), 10)

        pg.display.update()

thread_pygame = threading.Thread(target=pygame_window, args=())
thread_tkinter = threading.Thread(target=tkinter_window, args=())

thread_pygame.start()
# thread_tkinter.start()
