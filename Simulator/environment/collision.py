import numpy as np
# import torch
from numpy.linalg import inv, det
import copy
import time


def find_collision(target_lines, obstacles_lines):
    
    As, Bs = puck_lines_to_matrix(target_lines, obstacles_lines)


    # print(len(As))
    # cuda0 = torch.device('cuda:0')
    # As = torch.tensor(As,device=cuda0)
    # Bs = torch.tensor(Bs,device=cuda0)
    # # As_inv = torch.linalg.inv(As)
    # Xs = torch.linalg.solve(As, Bs)
    # Xs = Xs.detach().cpu().numpy().squeeze()

    Xs = np.linalg.solve(As, Bs)
    
    collision_points = []
    collision_dists = []
    n = len(obstacles_lines)
    for i in range(len(target_lines)):
        x01, x11, y01, y11 = get_line_points(target_lines[i])
        line1 = target_lines[i]
        t_ = np.array([Xs[i*n:(i+1)*n,0]])
        s_ = np.array([Xs[i*n:(i+1)*n,1]])
        # print(t_)
        params_t = np.where(((0 <= t_) & (t_ <= 1) & (0 <= s_) & (s_ <= 1)), t_, np.inf)
        params_t = params_t[np.isfinite(params_t)]
        points = np.vstack([(x11-x01) * params_t + x01, (y11-y01) * params_t + y01]).T

        collision_points.extend(points)


    collision = len(collision_points) > 0
    if collision:
        mean_point = np.mean(np.array(collision_points), axis=0)
    else:
        mean_point = [None, None]

    return collision, mean_point[0], mean_point[1]

def find_collision_rays(rays, obstacles_lines):
    
    # T0 = time.time_ns()
    As, Bs = puck_lines_to_matrix(rays, obstacles_lines)

    # T1 = time.time_ns()
    # cuda0 = torch.device('cuda:0')
    # As = torch.tensor(As,device=cuda0)
    # Bs = torch.tensor(Bs,device=cuda0)
    # As_inv = torch.linalg.inv(As)
    # Xs = torch.linalg.solve(As, Bs)
    # Xs = Xs.detach().cpu().numpy().squeeze()

    Xs = np.linalg.solve(As, Bs)
    # T2 = time.time_ns()

    scan_points = []
    scan_dists = []
    n = len(obstacles_lines)
    for i in range(len(rays)):
        x01, x11, y01, y11 = get_line_points(rays[i])
        line1 = rays[i]
        t_ = np.array([Xs[i*n:(i+1)*n,0]])
        s_ = np.array([Xs[i*n:(i+1)*n,1]])
        # print(t_)
        params_t = np.where(((0 <= t_) & (t_ <= 1) & (0 <= s_) & (s_ <= 1)), t_, np.inf)
        params_t = params_t[np.isfinite(params_t)]
        points = np.vstack([(x11-x01) * params_t + x01, (y11-y01) * params_t + y01]).T
        
        zero_points = np.full_like(points, line1[0])
        dists = np.linalg.norm(points - zero_points, axis=1)

        if len(dists) > 0:
            min_id = np.argmin(dists)
            point = points[min_id]
            dist = dists[min_id]
        else:
            point = copy.copy(line1[1])
            dist = np.inf
            dist = 500

        scan_points.append(point)
        scan_dists.append(dist)

    # T3 = time.time_ns()

    # print(f'puck - {(T1-T0) / 10**6} ms')
    # print(f'solve - {(T2-T1) / 10**6} ms')
    # print(f'unpuck - {(T3-T2) / 10**6} ms')

    return scan_points, scan_dists


def get_line_param(edge1, edge2):
    x0, y0 = edge1
    x1, y1 = edge2
    a = y1-y0
    b = x1-x0
    c = x0*(y1-y0) - y0*(x1-x0) 
    return a, b, c, x0, x1, y0, y1

def get_line_points(line):
    edge1, edge2 = line
    x0, y0 = edge1
    x1, y1 = edge2
    return x0, x1, y0, y1

def puck_lines_to_matrix(lines_1, lines_2):
    lines_1_vec = np.repeat(np.array(lines_1), len(lines_2), axis=0)
    lines_2_vec = np.array(lines_2 * len(lines_1))

    x01s = lines_1_vec[:,0,0]
    x11s = lines_1_vec[:,1,0]
    y01s = lines_1_vec[:,0,1]
    y11s = lines_1_vec[:,1,1]

    x02s = lines_2_vec[:,0,0]
    x12s = lines_2_vec[:,1,0]
    y02s = lines_2_vec[:,0,1]
    y12s = lines_2_vec[:,1,1]

    As = np.vstack([[(x11s-x01s), -(x12s-x02s)],[(y11s-y01s), -(y12s-y02s)]]).T.reshape((len(lines_1_vec), 2, 2))
    Bs = np.vstack([[(x02s-x01s)],[(y02s-y01s)]]).T.reshape((len(lines_1_vec), 2, 1))
    return As, Bs

def distance(point_1, point_2):
    return np.sqrt(np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]))