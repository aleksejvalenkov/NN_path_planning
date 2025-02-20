import numpy as np
# import torch
from numpy.linalg import inv, det
import copy
import time


def find_collision(target_lines, obstacles_lines):
    As = []
    Bs = []
    for target_line in target_lines:
        x01, x11, y01, y11 = get_line_points(target_line)
        for line_2 in obstacles_lines:
        
                x02, x12, y02, y12 = get_line_points(line_2)

                A = np.array([[(x11-x01), -(x12-x02)],
                            [(y11-y01), -(y12-y02)]])
                B = np.array([[(x02-x01)],
                            [(y02-y01)]])
                if (x11-x01)/(x12-x02) != (y11-y01)/(y12-y02):
                    As.append(A)
                    Bs.append(B)

    # print(len(As))
    # cuda0 = torch.device('cuda:0')
    # As = torch.tensor(As,device=cuda0)
    # Bs = torch.tensor(Bs,device=cuda0)
    # # As_inv = torch.linalg.inv(As)
    # Xs = torch.linalg.solve(As, Bs)
    # Xs = Xs.detach().cpu().numpy().squeeze()

    As = np.array(As)
    Bs = np.array(Bs)
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
    As = []
    Bs = []
    # T0 = time.time_ns()

    for ray in rays:
        x01, x11, y01, y11 = get_line_points(ray)
        for line_2 in obstacles_lines:
        
                x02, x12, y02, y12 = get_line_points(line_2)

                A = np.array([[(x11-x01), -(x12-x02)],
                            [(y11-y01), -(y12-y02)]])
                B = np.array([[(x02-x01)],
                            [(y02-y01)]])
                if (x11-x01)/(x12-x02) != (y11-y01)/(y12-y02):
                    As.append(A)
                    Bs.append(B)

    # print(len(As))
    # T1 = time.time_ns()
    # cuda0 = torch.device('cuda:0')
    # As = torch.tensor(As,device=cuda0)
    # Bs = torch.tensor(Bs,device=cuda0)
    # As_inv = torch.linalg.inv(As)
    # Xs = torch.linalg.solve(As, Bs)
    # Xs = Xs.detach().cpu().numpy().squeeze()

    As = np.array(As)
    Bs = np.array(Bs)
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

def distance(point_1, point_2):
    return np.sqrt(np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1]))