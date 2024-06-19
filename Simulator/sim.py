import pygame as pg
import sys
import random
from pygame.locals import *
from utils import *
from transforms import *

import matplotlib.pyplot as plt
import numpy as np
from kalman import tracker1

def move_robot(robot):
    global way_points

    # robot.teleop(teleop_vec=[1,0,0])
    s = 1
    d = 0
    if way_points:
        way_point = way_points[0]
        if way_point[0] - d <= robot.x <= way_point[0] + d and way_point[1] - d <= robot.y <= way_point[1] + d :
            if len(way_points) >= 1:
                way_points.pop(0)

            print('In goal calib point')
        x_s = constrain((way_point[0] - robot.x), -s, s)
        y_s = constrain((way_point[1] - robot.y), -s, s)

        robot.teleop(teleop_vec=[x_s, y_s, 0])

    pass

FPS = 30
silver = (194, 194, 194)
black = (0, 0, 0)
red =   (194,0 , 0)
green = (0, 194, 0)
blue =  (0, 0, 194)

WINDOW_SIZE = (1000, 1000)
robot_poses_truth_x = []
robot_poses_truth_y = []

robot_poses_est_x = []
robot_poses_est_y = []

robot_poses_pred_x = []
robot_poses_pred_y = []

robot_tracker = tracker1()

way_points = [[600,400], [600,600], [400,600], [400,400]]

std = 50

sensor_1_init_transform = np.array( 
        [[  1., 0., 500. ],
        [  0., 1., 20. ],
        [  0., 0., 1. ]]
        )
sensor_2_init_transform = np.array( 
        [[  1., 0., 20. ],
        [  0., 1., 980. ],
        [  0., 0., 1. ]]
        )
sensor_3_init_transform = np.array( 
        [[  1., 0., 960. ],
        [  0., 1., 960. ],
        [  0., 0., 1. ]]
        )

 
def main():
    recording = False
    calib_movement = False
    pg.init()
    screen = pg.display.set_mode(WINDOW_SIZE,RESIZABLE, 32)
    clock = pg.time.Clock()
    font = pg.font.SysFont("Ubuntu Condensed", 14, bold=False, italic=False)

    sensor_1_init_transform[0][2] += random.randint(-std//2, std//2)
    sensor_1_init_transform[1][2] += random.randint(-std//2, std//2)

    sensor_2_init_transform[0][2] += random.randint(-std//2, std//2)
    sensor_2_init_transform[1][2] += random.randint(-std//2, std//2)

    sensor_3_init_transform[0][2] += random.randint(-std//2, std//2)
    sensor_3_init_transform[1][2] += random.randint(-std//2, std//2)

    sensor_1 = UWBSensor(sensor_1_init_transform)
    sensor_2 = UWBSensor(sensor_2_init_transform)
    sensor_3 = UWBSensor(sensor_3_init_transform)
    # init objects
    robot = Robot(sensor_1, sensor_2, sensor_3)


    run = True
    while run:
        clock.tick(FPS)
        # Check events
        for i in pg.event.get():
            if i.type == QUIT:
                pg.quit()
                # sys.exit()
                run = False
                return
            elif i.type == KEYDOWN:
                # print(i.key)
                if i.key == 27:
                    pg.quit()
                    # sys.exit()
                    run = False
                    return
                if i.key == 13:
                    robot.auto_mode = not robot.auto_mode
                if i.key == 32:
                    recording = not recording
                if i.key == 112:
                    calib_movement = not calib_movement
                    
            elif i.type == pg.MOUSEBUTTONDOWN:
                if i.button == 1:
                    robot.way_point = i.pos

        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            robot.teleop(teleop_vec=[4,0,0])
        if keys[pg.K_s]:
            robot.teleop(teleop_vec=[-4,0,0])
        if keys[pg.K_a]:
            robot.teleop(teleop_vec=[0,0,-0.2])
        if keys[pg.K_d]:
            robot.teleop(teleop_vec=[0,0,0.2])



        # Render scene

        robot.update()
        if calib_movement:
            move_robot(robot)

        if recording:
            robot_poses_truth_x.append(robot.get_pose()[0])
            robot_poses_truth_y.append(robot.get_pose()[1])

            robot_poses_est_x.append(robot.est_robot_pose[0])
            robot_poses_est_y.append(robot.est_robot_pose[1])

            robot_poses_pred_x.append(robot.pred_robot_pose[0])
            robot_poses_pred_y.append(robot.pred_robot_pose[1])

        # Update display
        screen.fill(silver)

        robot.draw(screen)
        x,y,theta = robot.get_pose()
        text = font.render(f'Robot coordinates: x = {x:.2f}, y = {y:.2f}, theta = {theta:.2f}' , True, black)
        screen.blit(text, [10,10])
        text = font.render(f'Robot estimated coordinates: x = {robot.est_robot_pose[0]:.2f}, y = {robot.est_robot_pose[1]:.2f}, theta = {robot.est_robot_pose[2]:.2f}' , True, black)
        screen.blit(text, [10,25])
        if recording:
            pg.draw.circle(screen, red, (WINDOW_SIZE[0]-20, 20), 10)
        pg.display.update()


if __name__ == '__main__':
    main()

    # Some example data to display

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # fig.suptitle('Robot poses')
    # ax1.plot(robot_poses_truth_x, robot_poses_truth_y)
    # ax2.plot(robot_poses_est_x, robot_poses_est_y)
    # ax3.plot(robot_poses_pred_x, robot_poses_pred_y)

    plt.plot(robot_poses_truth_x, robot_poses_truth_y, label='Ground truth pose')
    plt.scatter(robot_poses_est_x, robot_poses_est_y, label='Sensor pose')
    plt.plot(robot_poses_pred_x, robot_poses_pred_y, label='Kalman pose')

    plt.legend(loc="lower right")
    plt.show()