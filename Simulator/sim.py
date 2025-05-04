import pygame as pg
import sys
from copy import deepcopy, copy
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
    s = 5
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
        robot.update()

def gen_robots(sensor_1_transform, sensor_2_transform, sensor_3_transform, std, x, y):
    sensor_1_transform = copy.copy(sensor_1_transform)
    sensor_2_transform = copy.copy(sensor_2_transform)
    sensor_3_transform = copy.copy(sensor_3_transform)

    dx_1 = np.random.normal(loc=0, scale=std, size=None)
    dy_1 = np.random.normal(loc=0, scale=std, size=None)
    dx_2 = np.random.normal(loc=0, scale=std, size=None)
    dy_2 = np.random.normal(loc=0, scale=std, size=None)
    dx_3 = np.random.normal(loc=0, scale=std, size=None)
    dy_3 = np.random.normal(loc=0, scale=std, size=None)

    sensor_1_transform[0][2] += dx_1
    sensor_1_transform[1][2] += dy_1

    sensor_2_transform[0][2] += dx_2
    sensor_2_transform[1][2] += dy_2

    sensor_3_transform[0][2] += dx_3
    sensor_3_transform[1][2] += dy_3

    sensor_1 = UWBSensor(sensor_1_transform)
    sensor_2 = UWBSensor(sensor_2_transform)
    sensor_3 = UWBSensor(sensor_3_transform)
    # init objects
    robot = Robot(sensor_1, sensor_2, sensor_3, x, y)
    # robot.update()

    return robot

def gen_particles(N, sensor_1_init_transform, sensor_2_init_transform, sensor_3_init_transform, std = 5, x=300, y=300):
    particles = []
    for i in range(N):
        particles.append(gen_robots(sensor_1_init_transform, sensor_2_init_transform, sensor_3_init_transform, std=std, x=x, y=y))
    return particles

def noise_transform(transform, std = 5):
    T = deepcopy(transform)
    error = np.random.normal(loc=0, scale=std, size=2)
    T[0][2] += error[0]
    T[1][2] += error[1]
    return T


def update_particle(particle, std = 5):
    tr1 = particle.sensor_1_nr.transform
    tr2 = particle.sensor_2_nr.transform
    tr3 = particle.sensor_3_nr.transform
    particle.sensor_1_nr.transform = noise_transform(tr1, std)
    particle.sensor_2_nr.transform = noise_transform(tr2, std)
    particle.sensor_3_nr.transform = noise_transform(tr3, std)

    # print(tr1 - particle.sensor_1_nr.transform)
    return particle

def gen_update_particles(particle, n, std = 5):
    particle_new = deepcopy(particle)
    # print('particle_new transform 1 = ', particle_new.sensor_1_nr.transform)
    # print('particle_new transform 2 = ', particle_new.sensor_2_nr.transform)
    # print('particle_new transform 3 = ', particle_new.sensor_3_nr.transform)
    particles = [particle_new]
    for i in range(n-1):
        particles.append(update_particle(deepcopy(particle), std=std))
        # print(particles[i].sensor_1_nr.transform)
    return particles

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

way_points = [[700,300], [700,700], [300,700], [300,300]]

# std = 50

dx = 200
dy = 100
mean_value = 0
standard_deviation = 100

pos_error_1 = np.random.normal(loc=mean_value, scale=standard_deviation, size=2)
pos_error_2 = np.random.normal(loc=mean_value, scale=standard_deviation, size=2)
pos_error_3 = np.random.normal(loc=mean_value, scale=standard_deviation, size=2)

sensor_1_init_transform = np.array( 
        [[  1., 0., 500. + pos_error_1[0] ],
        [  0., 1., 100. + pos_error_1[1] ],
        [  0., 0., 1. ]]
        )
sensor_2_init_transform = np.array( 
        [[  1., 0., 100. + pos_error_1[0] ],
        [  0., 1., 900. + pos_error_1[1] ],
        [  0., 0., 1. ]]
        )
sensor_3_init_transform = np.array( 
        [[  1., 0., 910. + pos_error_1[0] ],
        [  0., 1., 910. + pos_error_1[1] ],
        [  0., 0., 1. ]]
        )


 
def main():
    global way_points
    recording = False
    calib_movement = False
    calib_movement_2 = False
    pg.init()
    screen = pg.display.set_mode(WINDOW_SIZE, RESIZABLE, 32)
    clock = pg.time.Clock()
    font = pg.font.SysFont("Ubuntu Condensed", 14, bold=False, italic=False)

    print(f'Sensor 1 init at: \n{sensor_1_init_transform}')
    print(f'Sensor 2 init at: \n{sensor_2_init_transform}')
    print(f'Sensor 3 init at: \n{sensor_3_init_transform}')

    sensor_1 = UWBSensor(sensor_1_init_transform)
    sensor_2 = UWBSensor(sensor_2_init_transform)
    sensor_3 = UWBSensor(sensor_3_init_transform)
    # init objects
    robot_main = Robot(sensor_1, sensor_2, sensor_3, x=700, y=700)

    robot_particles = gen_particles(500, sensor_1_init_transform, sensor_2_init_transform, sensor_3_init_transform, std=10, x=300, y=300)

    run = True

    step = 0
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
                if i.key == 27 or i.key == 113:
                    pg.quit()
                    # sys.exit()
                    run = False
                    return
                if i.key == 13:
                    robot_main.auto_mode = not robot_main.auto_mode
                if i.key == 32:
                    recording = not recording
                if i.key == 112:
                    calib_movement = not calib_movement
                    
            elif i.type == pg.MOUSEBUTTONDOWN:
                if i.button == 1:
                    robot_main.way_point = i.pos

        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            robot_main.teleop(teleop_vec=[4,0,0])
        if keys[pg.K_s]:
            robot_main.teleop(teleop_vec=[-4,0,0])
        if keys[pg.K_a]:
            robot_main.teleop(teleop_vec=[0,0,-0.2])
        if keys[pg.K_d]:
            robot_main.teleop(teleop_vec=[0,0,0.2])



        # Render scene
        robot_main.update()

        if calib_movement:
            if way_points:
                measurements = []
                for robot in robot_particles:
                    move_robot(robot)
                    measurements.append(distance(robot.get_pose(), robot.est_robot_pose))
                    # print(robot.est_robot_pose)
                MAX = np.max(measurements)
                # measurements = np.nan_to_num(measurements, nan=MAX)
                particles_step = []

                
                rand_list = []
                for i in range(len(measurements)):
                    n = MAX // measurements[i]
                    print("MAX =", MAX, "n =",n)
                    rand_list.extend([robot_particles[i]]*int(n))

                print(len(rand_list))
                # for particle in rand_list:
                #     print(particle.sensor_1_nr.transform[0][2])
                

                N = 10
                for i in range(N):
                    k = random.randint(0, len(rand_list)-1)
                    # print(k)
                    index = rand_list[k]
                    # robot_main.sensor_1_nr = UWBSensor(particles[0].sensor_1_nr.transform)
                    # robot_main.sensor_2_nr = UWBSensor(particles[0].sensor_2_nr.transform)
                    # robot_main.sensor_3_nr = UWBSensor(particles[0].sensor_3_nr.transform)
                    # robot_main.update()

                    # particles_new = gen_particles((len(robot_particles)//N)-1, 
                    #                                     robot_particles[index].sensor_1_nr.transform, 
                    #                                     robot_particles[index].sensor_2_nr.transform, 
                    #                                     robot_particles[index].sensor_3_nr.transform,
                    #                                     std = measurements[index],
                    #                                     x = robot_particles[index].x, 
                    #                                     y = robot_particles[index].y )

                    particle = rand_list[k]
                    particles_new = gen_update_particles(particle, n=(len(robot_particles)//N), std=50) 
                    
                    # print(particles[index].x, particles[index].y)
                    # particles_new.append(robot_particles[index])

                    # print('TEAR', particles_new[0].sensor_1_nr.transform)  
                    particles_step.extend(particles_new)

                    # for robot in particles_step:
                    #     robot.update()


                # print(len(particles_step))
                robot_particles = particles_step
                # for robot in robot_particles:
                #     robot.update()
            else:
                calib_movement = False
                calib_movement_2 = True
                way_points = [[700,300], [700,700], [300,700], [300,300]]

        if calib_movement_2:
            if way_points:
                for robot in robot_particles:
                    move_robot(robot)
                    robot.errors.append(distance(robot.get_pose(), robot.est_robot_pose))
                    # print(distance(robot.get_pose(), robot.est_robot_pose))
            else:
                norms = []
                for robot in robot_particles:
                    errors = np.array(robot.errors)
                    n = np.std(errors)
                    norms.append(n)
                    # print(n)
                if norms:
                    # print(norms)
                    norms = np.nan_to_num(norms, nan=1000.)
                    # print(norms)
                    h = np.array(norms)
                    print(np.argmin(h), np.min(h))
                    id = np.argmin(h)

                    print(f'Sensor 1 calib at: \n{robot_particles[id].sensor_1_nr.transform}')
                    print(f'Sensor 2 calib at: \n{robot_particles[id].sensor_2_nr.transform}')
                    print(f'Sensor 3 calib at: \n{robot_particles[id].sensor_3_nr.transform}')
                    robot_main.sensor_1_nr = UWBSensor(robot_particles[id].sensor_1_nr.transform)
                    robot_main.sensor_2_nr = UWBSensor(robot_particles[id].sensor_2_nr.transform)
                    robot_main.sensor_3_nr = UWBSensor(robot_particles[id].sensor_3_nr.transform)
                    norms = []
                    particles = []
                    calib_movement_2 = False


        for robot in robot_particles:
            robot.update()

        if recording:
            robot_poses_truth_x.append(robot_main.get_pose()[0])
            robot_poses_truth_y.append(robot_main.get_pose()[1])

            robot_poses_est_x.append(robot_main.est_robot_pose[0])
            robot_poses_est_y.append(robot_main.est_robot_pose[1])

            robot_poses_pred_x.append(robot_main.pred_robot_pose[0])
            robot_poses_pred_y.append(robot_main.pred_robot_pose[1])
        
        step += 1
        # Update display
        screen.fill(silver)

        robot_main.draw(screen)

        for robot in robot_particles:
            robot.draw(screen)

        x,y,theta = robot_main.get_pose()
        text = font.render(f'Robot coordinates: x = {x:.2f}, y = {y:.2f}, theta = {theta:.2f}' , True, black)
        screen.blit(text, [10,10])
        text = font.render(f'Robot estimated coordinates: x = {robot_main.est_robot_pose[0]:.2f}, y = {robot_main.est_robot_pose[1]:.2f}, theta = {robot_main.est_robot_pose[2]:.2f}' , True, black)
        screen.blit(text, [10,25])
        if recording:
            pg.draw.circle(screen, red, (WINDOW_SIZE[0]-20, 20), 10)
            [600,400], [600,600], [400,600], [400,400]
            pg.draw.aaline(screen, blue, [600,400], [600,600])
            pg.draw.aaline(screen, blue, [600,600], [400,600])
            pg.draw.aaline(screen, blue, [400,600], [400,400])
            pg.draw.aaline(screen, blue, [400,400], [600,400])
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