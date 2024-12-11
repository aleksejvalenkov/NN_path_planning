import numpy as np
from numpy.linalg import inv, det

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


def lines_intersection(line1, line2):
    intersection = False
    x = None
    y = None
    a1, b1, c1, x01, x11, y01, y11 = get_line_param(line1[0], line1[1])
    a2, b2, c2, x02, x12, y02, y12 = get_line_param(line2[0], line2[1])

    A = np.array([[a1, b1],[a2, b2]])
    B = np.array([[c1],[c2]])
    if det(A) != 0.0:
        X = (inv(A) @ B).T
    else:
        return intersection, x, y
    
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
        x = None
        y = None
    return intersection, x, y

def get_line_param(edge1, edge2):
    x0, y0 = edge1
    x1, y1 = edge2
    a = y1-y0
    b = x1-x0
    c = x0*(y1-y0) - y0*(x1-x0) 
    return a, b, c, x0, x1, y0, y1