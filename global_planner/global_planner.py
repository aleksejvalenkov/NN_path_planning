from global_planner.a_star import AStar
from global_planner.a_star_s import AStarPlanner
import copy
import cv2 
import numpy as np

METRIC_KF = 100 # 1m = 100px

class GlobalPlanner:
    def __init__(self):
        pass
        self.occupation_range = 60
        self.grid_size = 30 # In px
        self.bin_map = None
        self.bin_map_occupation = None
        self.bin_map_decomposition = None
        self.load_map()
        self.prepare_map()

              
        self.sorver = AStar(self.get_bin_map_decomposition())
        ox = []
        oy = []
        bin_map_decomposition = self.get_bin_map_decomposition()
        for i, j in np.ndindex(bin_map_decomposition.shape):
            if bin_map_decomposition[i,j] == 1:
                ox.append(i)
                oy.append(j)

        self.sorver_2 = AStarPlanner(ox, oy, 1, 0.01)

        
    
    def prepare_map(self):
        rows, cols = self.bin_map_occupation.shape
        # print('rows = ', rows, 'cols = ', cols)
        nx, ny = (cols//self.grid_size, rows//self.grid_size)
        x = np.linspace(0, cols-1, nx+1)
        y = np.linspace(0, rows-1, ny+1)
        xv, yv = np.meshgrid(x, y)
        empty_map = np.zeros(((rows//self.grid_size)+1, (cols//self.grid_size)+1))
        # print('empty_map = ', empty_map.shape)
        # print('xv = ', xv.shape)
        # print('yv = ', yv.shape)
        self.bin_map_decomposition = np.stack((xv, yv, empty_map), axis=2)
        # print('bin_map_decomposition = ', self.bin_map_decomposition.shape)
        # print('bin_map_occupation = ', self.bin_map_occupation.shape)
        for row, col, p in np.ndindex(self.bin_map_decomposition.shape):
            # print('row = ', row, 'col = ', col)
            x = int(self.bin_map_decomposition[row, col][0])
            y = int(self.bin_map_decomposition[row, col][1])
            # print('x = ', x, 'y = ', y)
            if self.bin_map_occupation[y, x] == 1:
                self.bin_map_decomposition[row, col, 2] = 1

        # print('bin_map_decomposition = ', self.bin_map_decomposition)

    def load_map(self):
        map = cv2.imread("global_planner/map/map_bin.jpg")
        # print(map)
        gray_map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((self.occupation_range,self.occupation_range),np.float32)/25
        gray_map_blur = cv2.filter2D(gray_map,-1,kernel)
        self.bin_map = np.where(gray_map > 127, 1, 0)
        self.bin_map_occupation = np.where(gray_map_blur > 127, 1, 0)
        cv2.imwrite("global_planner/map/map_bin_ext.jpg", self.bin_map_occupation*255)

    def plan_path(self, start, goal):
        # print(self.bin_map_decomposition.shape)
        start_glolbal = self.convert_to_global_map_coords(start[:2])
        goal_glolbal = self.convert_to_global_map_coords(goal[:2])
        # print('start = ', start_glolbal, 'goal = ', goal_glolbal)
        # path = self.sorver.find_path(start_glolbal, goal_glolbal)
        path_x, path_y = self.sorver_2.planning(start_glolbal[0],start_glolbal[1],goal_glolbal[0],goal_glolbal[1])
        path2 = np.stack((np.array(path_x), np.array(path_y)), axis=-1)
        path2 = np.flip(path2, 0)
        # print(path, path2)
        path = self.convert_path_to_map_coords(path2)
        if path:
            path = np.array(path)
            path = np.hstack((path, np.ones((len(path),1))*goal[2]))
            # path = np.vstack((path, np.array(goal)))
        # path.append(goal)
        # path = np.array(path)
        # Remove straight sections from the path
        if path is not None and len(path) > 2:
            filtered_path = [path[0]]
            for i in range(1, len(path) - 1):
                prev = path[i - 1]
                curr = path[i]
                next = path[i + 1]
                # Check if three points are collinear (straight line)
                v1 = curr[:2] - prev[:2]
                v2 = next[:2] - curr[:2]
                if not np.allclose(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)):
                    filtered_path.append(curr)
            filtered_path.append(path[-1])
            path = np.array(filtered_path)
        return path[1:]

    def get_bin_map(self):
        return self.bin_map
    
    def get_bin_map_occupation(self):
        return self.bin_map_occupation
    
    def get_bin_map_decomposition(self):
        return self.bin_map_decomposition[:, :, 2]
    
    def convert_to_global_map_coords(self, point):
        # print(self.bin_map_decomposition[:, :, :2])
        min_range_point = np.inf
        min_range_point_coords = None
        for row, col, p in np.ndindex(self.bin_map_decomposition[:, :, :2].shape):
            # print(self.bin_map_decomposition[:, :, :2][row, col])
            range_point = np.linalg.norm(self.bin_map_decomposition[:, :, :2][row, col] - point)
            if range_point < min_range_point:
                min_range_point = range_point
                min_range_point_coords = (row, col)

        return min_range_point_coords

    def convert_path_to_map_coords(self, path):
        path_map = []
        if path is not None:
            for x, y in path:
                path_map.append(tuple(self.bin_map_decomposition[:, :, :2][x, y]))
            return path_map
        else:
            return None
