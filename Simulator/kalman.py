from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from kf_book.book_plots import plot_filter
from kf_book.book_plots import plot_measurements
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn

class PosSensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        
    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]

R_std = 5
Q_std = 5

def tracker1():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 0.25   # time step
    s = 1
    tracker.F = np.array([[s, dt, 0,  0],
                          [0,  s, 0,  0],
                          [0,  0, s, dt],
                          [0,  0, 0,  s]])
    tracker.u = 0.
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]])

    tracker.R = np.eye(2) * R_std
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std)
    tracker.Q = block_diag(q, q)

    tracker.x = np.array([[100, 0, 100, 0]]).T
    tracker.P = np.eye(4) * 5000.
    return tracker


def get_predict(pose):
    # simulate robot movement
    # N = 30
    # sensor = PosSensor((0, 0), (2, .2), noise_std=R_std)
    # print(sensor.read())
    # # zs = np.array([sensor.read() for _ in range(N)])

    # run filter
    robot_tracker = tracker1()
    mu, cov, _, _ = robot_tracker.update(pose)

    mean = (x[0, 0], x[2, 0])

        
