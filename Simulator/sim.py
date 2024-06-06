import pygame as pg
import sys
from pygame.locals import *
from utils import *
from transforms import *


FPS = 30
silver = (194, 194, 194)
black = (0, 0, 0)
red =   (194,0 , 0)
green = (0, 194, 0)
blue =  (0, 0, 194)

WINDOW_SIZE = (1000, 1000)
 
pg.init()
screen = pg.display.set_mode(WINDOW_SIZE,RESIZABLE, 32)
clock = pg.time.Clock()
font = pg.font.SysFont("Ubuntu Condensed", 14, bold=False, italic=False)


# init objects
robot = Robot()
 
while True:
    clock.tick(FPS)
# Check events
    for i in pg.event.get():
        if i.type == QUIT:
            pg.quit()
            sys.exit()
        elif i.type == KEYDOWN:
            if i.key == 27:
                pg.quit()
                sys.exit()

    keys = pg.key.get_pressed()
    if keys[pg.K_w]:
        robot.teleop(teleop_vec=[0,-2])
    if keys[pg.K_s]:
        robot.teleop(teleop_vec=[0,2])
    if keys[pg.K_a]:
        robot.teleop(teleop_vec=[-2,0])
    if keys[pg.K_d]:
        robot.teleop(teleop_vec=[2,0])



# Render scene

    robot.update()

# Update display
    screen.fill(silver)

    robot.draw(screen)
    x,y = robot.get_pose()
    text = font.render(f'Robot coordinates: x = {x:.2f}, y = {y:.2f}' , True, black)
    screen.blit(text, [10,10])
    pg.display.update()