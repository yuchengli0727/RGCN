import matplotlib.pyplot as plt
import numpy as np


'''
ac = np.load("PeMS04.npz")
data = ac["data"]
flow = data.transpose([1, 0, 2])[:, :, 0]
speed = data.transpose([1, 0, 2])[:, :, 1]
occup = data.transpose([1, 0, 2])[:, :, 2]

flow_x = flow[1, 0:1440]
speed = speed[1, 0:1440]
occup = occup[1, 0:1440]
plt.plot(flow_x)
plt.plot(occup)
plt.plot(speed)

plt.xlabel("time")
plt.ylabel("value")
plt.legend(["traffic flow", "speed", "occupancy"], loc="upper right")
plt.show()
'''
col1 = 'steelblue'
col2 = 'red'
x = [1, 2, 3, 4, 5]
x_ticks = np.arange(1, 6, 1)
'''
mae = [27.44, 25.79, 25.13, 26.41, 26.90]
rmse = [39.35, 37.71, 36, 37.32, 37.11]
'''
mae = [34.14, 29.79, 28.13, 29.31, 31.90]
rmse = [49.35, 46.71, 43, 44.32, 45.11]

fig,ax = plt.subplots()
ax.plot(x, mae, color=col1, marker='o', linewidth=3)
ax.set_ylabel('MAE')
ax.set_xlabel("图卷积阶数")
#define second y-axis that shares x-axis with current plot
ax2 = ax.twinx()
#add second line to plot
ax2.plot(x, rmse, color=col2, marker='o', linewidth=3)
#add second y-axis label
ax2.set_ylabel('RMSE')
plt.xticks(x_ticks)
plt.show()

plt.plot(x, mae, color=col1, marker='o', linewidth=3)
plt.plot(x, rmse, color=col2, marker='o', linewidth=3)
plt.legend(["MAE", "RMSE"], loc="upper right")
plt.show()
