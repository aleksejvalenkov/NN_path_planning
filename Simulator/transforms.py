import numpy as np
from scipy.spatial.transform import Rotation as R

def get_transform(t_vec, theta):
    transform = np.eye(3)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])

    transform[:2, :2] = rot_matrix
    transform[:2, 2] = t_vec.T

    return transform

def get_XYTheta(transform):
    x = transform[0,2]
    y = transform[1,2]
    theta = np.arctan2(transform[1,0], transform[0,0])

    return x, y, theta

def constrain(x, a, b):
    if x > 0.0 and x < abs(a):
        return a
    elif x > 0.0 and x > abs(b):
        return b
    elif x < 0.0 and x > -abs(a):
        return -a
    elif x < 0.0 and x < -abs(b):
        return -b
    else:
        return x
    
def distance(p1, p2):
    d = np.sqrt(pow((p1[0] - p2[0]), 2) + pow((p1[1] - p2[1]), 2))
    d = np.random.normal(loc=d, scale=10, size=None)
    return d

def point_form_two_rounds(c1, d1, c2, d2):
    dist = distance(c1, c2)
    if dist > d1 + d2:
        op = (dist / (d1 + d2)) + 0.1
        d1 = d1 * op
        d2 = d2 * op


    x = c1[0] + (c2[0] - c1[0])*((d1**2 - d2**2 + dist**2)/(2*dist**2))
    k = (c2[1]-c1[1])/(c2[0]-c1[0])
    y = c1[1] + (k * (x - c1[0]))
    return [x, y]

def line_from_vec_and_point(vec, p):
    A = vec[1]/vec[0]
    B = -1
    C = p[1] - A * p[0]
    return np.array((A, B, C))

def line_from_two_point(p1, p2):
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0]*p2[1] - p2[0]*p1[1] 
    return np.array((A, B, C))

def point_from_two_lines(L1, L2):
    A = np.array([L1[:2], L2[:2]])
    B = np.array([-L1[2], -L2[2]]).T
    X = np.linalg.inv(A) @ B
    return X.T
