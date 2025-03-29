import numpy as np
from scipy.spatial.transform import Rotation as R

def get_transform(t_vec, theta):
    transform = np.eye(3)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])

    transform[:2, :2] = rot_matrix
    transform[:2, 2] = t_vec.T

    return transform

def get_rot_mat_from_theta(theta):
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return rot_matrix

def get_theta_from_rot_mat(rot_matrix):
    theta = np.arctan2(rot_matrix[1,0], rot_matrix[0,0])
    return theta

def get_theta_error(theta_1, theta_2):
    mat_1 = get_rot_mat_from_theta(theta_1)
    mat_2 = get_rot_mat_from_theta(theta_2)
    mat_err = mat_2 @ mat_1.T
    return get_theta_from_rot_mat(mat_err)

def get_target_angle(robot_point, target_point):
    b = target_point[0] - robot_point[0]
    a = target_point[1] - robot_point[1]
    return np.arctan2(a,b)


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