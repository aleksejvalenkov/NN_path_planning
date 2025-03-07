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

def constrain(x, min_x, max_x):
    if x > max_x:
        return max_x
    elif x < min_x:
        return min_x
    else:
        return x