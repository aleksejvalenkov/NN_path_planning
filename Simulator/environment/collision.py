import numpy as np
from numpy.linalg import inv, det
import copy


def find_collision(BB1, BB2):

    collision = False
    points = []
    
    for point1_index in range(len(BB1)):
        for point2_index in range(len(BB2)):
            if point1_index == len(BB1)-1:
                line1 = [BB1[point1_index], BB1[0]]
            else:
                line1 = [BB1[point1_index], BB1[point1_index+1]]

            if point2_index == len(BB2)-1:
                line2 = [BB2[point2_index], BB2[0]]
            else:
                line2 = [BB2[point2_index], BB2[point2_index+1]]
            
            intersection, x, y = lines_intersection(line1, line2)
            if intersection:
                collision = True
                points.append([x, y])

    if collision:
        mean_point = np.mean(np.array(points), axis=0)
    else:
        mean_point = [None, None]

    return collision, mean_point[0], mean_point[1]

def find_collision_rays(line1, BB2):

    point = copy.copy(line1[1])
    dist_min = np.inf
    i_min = -1
    i = 0
    
    for point2_index in range(len(BB2)):

        if point2_index == len(BB2)-1:
            line2 = [BB2[point2_index], BB2[0]]
        else:
            line2 = [BB2[point2_index], BB2[point2_index+1]]
        
        intersection, x, y = lines_intersection(line1, line2)
        # print('Координаты точки ', x, y)
        if intersection:
        
            dist = distanse(line1[0], [x, y])
            if dist <= dist_min:
                dist_min = dist
                point = [x, y]
                i_min = i
        i+=1
    # if i_min != -1:
        
        # print(f'Итог расстояние {dist_min}')
        # print(f'Пересечение луча и линии {i_min}')
    return point


def lines_intersection(line1, line2):
    intersection = False
    x = None
    y = None
    a1, b1, c1, x01, x11, y01, y11 = get_line_param(line1[0], line1[1])
    a2, b2, c2, x02, x12, y02, y12 = get_line_param(line2[0], line2[1])

    A = np.array([[a1, b1],[a2, b2]])
    B = np.array([[c1],[c2]])
    deter = a1 * b2 + b1 * a2
    if det(A) != 0.0:
        X = np.dot(inv(A) , B).T
    else:
        print("paralel lines")
        return intersection, x, y
    
    # x = np.round(X[0][0])
    # y = np.round(-X[0][1])

    x = X[0][0]
    y = -X[0][1]

    if (x01 <= x <= x11 \
        or x01 >= x >= x11) \
        and (y01 <= y <= y11 \
        or y01 >= y >= y11) \
        and (x02 <= x <= x12 \
        or x02 >= x >= x12) \
        and (y02 <= y <= y12 \
        or y02 >= y >= y12):
        intersection = True
    else:
        intersection = False
        x = x
        y = y
    return intersection, x, y

def get_line_param(edge1, edge2):
    x0, y0 = edge1
    x1, y1 = edge2
    a = y1-y0
    b = x1-x0
    c = x0*(y1-y0) - y0*(x1-x0) 
    return a, b, c, x0, x1, y0, y1

def distanse(point_1, point_2):
    return np.sqrt(np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]))