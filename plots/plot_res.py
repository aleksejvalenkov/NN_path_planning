import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.ndimage import uniform_filter1d

window = 30
length = 1000000
# headers = ["Wall time", "Step", "Value"]
df_DDPG_r1 = pd.read_csv('plots/DDPG-r1-25-03-13_09-22-32-045596_PPO.csv')

df_PP0_r1_1 = pd.read_csv('plots/PP0-r1-1-25-03-22_18-40-28-188343_PPO.csv')
df_PP0_r1_2 = pd.read_csv('plots/PP0-r1-2-25-03-23_00-33-21-627698_PPO.csv')

df_PP0_r2 = pd.read_csv('plots/PP0-r2-25-03-26_09-32-35-840813_PPO.csv')
# print (df)

x_DDPG_r1 = df_DDPG_r1['Step'].to_numpy()
y_DDPG_r1 = df_DDPG_r1['Value'].to_numpy()/10000
x_DDPG_r1 = x_DDPG_r1[x_DDPG_r1 < length]
y_DDPG_r1 = y_DDPG_r1[0:len(x_DDPG_r1)]
y_DDPG_r1_smoothed = uniform_filter1d(y_DDPG_r1, size=window)


x_PP0_r1_1 = df_PP0_r1_1['Step'].to_numpy()
y_PP0_r1_1 = df_PP0_r1_1['Value'].to_numpy()
x_PP0_r1_2 = df_PP0_r1_2['Step'].to_numpy()+x_PP0_r1_1[-1]
y_PP0_r1_2 = df_PP0_r1_2['Value'].to_numpy()
x_PP0_r1 = np.hstack((x_PP0_r1_1, x_PP0_r1_2))
y_PP0_r1 = np.hstack((y_PP0_r1_1, y_PP0_r1_2))
x_PP0_r1 = x_PP0_r1[x_PP0_r1 < length]
y_PP0_r1 = y_PP0_r1[0:len(x_PP0_r1)]
y_PP0_r1_smoothed = uniform_filter1d(y_PP0_r1, size=window)

x_PP0_r2 = df_PP0_r2['Step'].to_numpy()
y_PP0_r2 = df_PP0_r2['Value'].to_numpy()/1000
x_PP0_r2 = x_PP0_r2[x_PP0_r2 < length]
y_PP0_r2 = y_PP0_r2[0:len(x_PP0_r2)]
y_PP0_r2_smoothed = uniform_filter1d(y_PP0_r2, size=window)


# plot
fig, ax = plt.subplots()

ax.plot(x_DDPG_r1, y_DDPG_r1_smoothed, label='DDPG-r1')
ax.plot(x_PP0_r1, y_PP0_r1_smoothed, label='PP0-r1')
ax.plot(x_PP0_r2, y_PP0_r2_smoothed, label='PP0-r2')

ax.legend()
ax.grid(True)

plt.show()