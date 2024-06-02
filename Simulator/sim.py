import pygame as pg
import sys
from pygame.locals import *
from utils import *
 
FPS = 30
silver = (194, 194, 194)
black = (0, 0, 0)
red =   (194,0 , 0)
green = (0, 194, 0)
blue =  (0, 0, 194)

WINDOW_SIZE = (600, 600)
 
pg.init()
screen = pg.display.set_mode(WINDOW_SIZE,RESIZABLE, 32)
clock = pg.time.Clock()
font = pg.font.SysFont("Ubuntu Condensed", 35, bold=False, italic=False)
year = 2021
 
while True:
    clock.tick(FPS)
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



# Render scene


# Update display
    screen.fill(silver)


    # text = font.render("Привет Мир! "+str(year)+" год!",True,silver)
    # screen.blit(text, [150,170])
    pg.display.update()