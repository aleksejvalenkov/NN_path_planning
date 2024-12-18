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

def get_XY(transform):
    x = transform[0,2]
    y = transform[1,2]
    return x, y

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