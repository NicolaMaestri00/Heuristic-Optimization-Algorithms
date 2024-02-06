import matplotlib.pyplot as plt
import numpy as np
from pylab import figure, cm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import random

fig = plt.figure()
gif_title = 'PSO_x10.gif'

def fitness(x, y):
  return (np.sin(x)*np.sin(y)+x/7)*np.exp(-(x**2+y**2)/50)

# def fitness(x, y):
#   return -(x**2+y**2)

# Points
x_points = [-4, 8, 2, -2]
y_points = [-4, -5, 6, 2]
# x_points = [random.randint(-10, 10) for _ in range(4)]
# y_points = [random.randint(-10, 10) for _ in range(4)]

# Velocities
x_vel = [-0.3, 0.0, 0.3, 0.0]
y_vel = [-0.3, 0.3, 0.0, 0.0]

# Parameters
w = 0.95
phi1 = 0.15
phi2 = 0.08
N = 100

# initialization of best positions so far
x_p = [-4, 8, 2, -2]
y_p = [-4, -5, 6, 2]

f1_best = fitness(x_p[0], y_p[0])
f2_best = fitness(x_p[1], y_p[1])
f3_best = fitness(x_p[2], y_p[2])
f4_best = fitness(x_p[3], y_p[3])

f_best = f1_best
x_best = x_p[0]
y_best = y_p[0]

if f_best<f2_best:
  f_best = f2_best
  x_best = x_p[1]
  y_best = y_p[1]
if f_best<f3_best:
  f_best = f3_best
  x_best = x_p[2]
  y_best = y_p[2]
if f_best<f4_best:
  f_best = f4_best
  x_best = x_p[3]
  y_best = y_p[3]

#-------------------------------------------------------------------------------
x1_min = -10.0
x1_max =  10.0
x2_min = -10.0
x2_max =  10.0

x_ax, y_ax = np.meshgrid(np.arange(x1_min,x1_max, 0.1), np.arange(x2_min,x2_max, 0.1))
y = fitness(x_ax, y_ax)

plt.imshow(y,extent=[x1_min,x1_max,x2_min,x2_max], cmap=cm.jet, origin='lower')
plt.colorbar()

plt.scatter(x_points, y_points, color='blue', marker='o', label='Points')
# Annotate points with arrows
for x, y in zip(x_points, y_points):
    plt.annotate(f'({round(x,2)}, {round(y,2)})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# Draw arrows x1
arrow_start = (x_points[0], y_points[0])
arrow_end = (x_points[0]+x_vel[0], y_points[0]+y_vel[0])
plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
          color='red', width=0.05, head_width=0.2, head_length=0.2, length_includes_head=True)

# Draw arrows x2
arrow_start = (x_points[1], y_points[1])
arrow_end = (x_points[1]+x_vel[1], y_points[1]+y_vel[1])
plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
          color='red', width=0.05, head_width=0.2, head_length=0.2, length_includes_head=True)

# Draw arrows x3
arrow_start = (x_points[2], y_points[2])
arrow_end = (x_points[2]+x_vel[2], y_points[2]+y_vel[2])
plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
          color='red', width=0.05, head_width=0.2, head_length=0.2, length_includes_head=True)

# Draw arrows x4
arrow_start = (x_points[3], y_points[3])
arrow_end = (x_points[3]+x_vel[3], y_points[3]+y_vel[3])
plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
          color='red', width=0.05, head_width=0.2, head_length=0.2, length_includes_head=True)

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Particle Swarm Optimization')

#-------------------------------------------------------------------------------
metadata = dict(title='Movie', artist='codinglikemad')
writer = PillowWriter(fps=15, metadata=metadata)

with writer.saving(fig, gif_title, 100):
    for i in range(N):
        # New velocities
        x_vel[0] = w*x_vel[0] + phi1*(x_p[0]- x_points[0]) + phi2*(x_best - x_points[0])
        y_vel[0] = w*y_vel[0] + phi1*(y_p[0]- y_points[0]) + phi2*(y_best - y_points[0])

        x_vel[1] = w*x_vel[1] + phi1*(x_p[1]- x_points[1]) + phi2*(x_best - x_points[1])
        y_vel[1] = w*y_vel[1] + phi1*(y_p[1]- y_points[1]) + phi2*(y_best - y_points[1])

        x_vel[2] = w*x_vel[2] + phi1*(x_p[2]- x_points[2]) + phi2*(x_best - x_points[2])
        y_vel[2] = w*y_vel[2] + phi1*(y_p[2]- y_points[2]) + phi2*(y_best - y_points[2])

        x_vel[3] = w*x_vel[3] + phi1*(x_p[3]- x_points[3]) + phi2*(x_best - x_points[3])
        y_vel[3] = w*y_vel[3] + phi1*(y_p[3]- y_points[3]) + phi2*(y_best - y_points[3])

        # New positions
        x_points[0] = x_points[0] + x_vel[0]
        y_points[0] = y_points[0] + y_vel[0]

        x_points[1] = x_points[1] + x_vel[1]
        y_points[1] = y_points[1] + y_vel[1]

        x_points[2] = x_points[2] + x_vel[2]
        y_points[2] = y_points[2] + y_vel[2]

        x_points[3] = x_points[3] + x_vel[3]
        y_points[3] = y_points[3] + y_vel[3]

        # Update best position so far
        f_x1 = fitness(x_points[0], y_points[0])
        f_x2 = fitness(x_points[1], y_points[1])
        f_x3 = fitness(x_points[2], y_points[2])
        f_x4 = fitness(x_points[3], y_points[3])

        if f1_best < f_x1:
            f1_best = f_x1
            x_p[0] = x_points[0]
            y_p[0] = y_points[0]
            if f_best < f_x1:
                f_best = f_x1
                x_best = x_points[0]
                y_best = y_points[0]

        if f2_best < f_x2:
            f2_best = f_x2
            x_p[1] = x_points[1]
            y_p[1] = y_points[1]
            if f_best < f_x2:
                f_best = f_x2
                x_best = x_points[1]
                y_best = y_points[1]

        if f3_best < f_x3:
            f3_best = f_x3
            x_p[2] = x_points[2]
            y_p[2] = y_points[2]
            if f_best < f_x3:
                f_best = f_x3
                x_best = x_points[2]
                y_best = y_points[2]

        if f4_best < f_x4:
            f4_best = f_x4
            x_p[3] = x_points[3]
            y_p[3] = y_points[3]
            if f_best < f_x4:
                f_best = f_x4
                x_best = x_points[3]
                y_best = y_points[3]

        plt.scatter(x_points[0], y_points[0], color='blue', marker='o', label='Points')

        # Draw arrows x1
        arrow_start = (x_points[0], y_points[0])
        arrow_end = (x_points[0]+x_vel[0], y_points[0]+y_vel[0])
        plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
                    color='red', width=0.05, head_width=0.2, head_length=0.2, length_includes_head=True)

        # # Draw arrows x2
        # arrow_start = (x_points[1], y_points[1])
        # arrow_end = (x_points[1]+x_vel[1], y_points[1]+y_vel[1])
        # plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
        #             color='red', width=0.05, head_width=0.2, head_length=0.2, length_includes_head=True)

        # # Draw arrows x3
        # arrow_start = (x_points[2], y_points[2])
        # arrow_end = (x_points[2]+x_vel[2], y_points[2]+y_vel[2])
        # plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
        #             color='red', width=0.05, head_width=0.2, head_length=0.2, length_includes_head=True)

        # # Draw arrows x4
        # arrow_start = (x_points[3], y_points[3])
        # arrow_end = (x_points[3]+x_vel[3], y_points[3]+y_vel[3])
        # plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1],
        #             color='red', width=0.05, head_width=0.2, head_length=0.2, length_includes_head=True)

        # Set labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        writer.grab_frame()