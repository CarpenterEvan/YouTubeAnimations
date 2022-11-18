import pickle
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('dark_background')


def format():
    ax[1][1].set_title("Difference")
    ax[1][1].set_xlabel("time")
    ax[1][1].set_ylabel("distance to origin")
    ax[1][1].set_xlim([0, 40])
    ax[1][1].set_ylim([0, 50])

    yz_ax.set_title("Looking head on")
    yz_ax.set_xlabel("y")
    yz_ax.set_ylabel("z")
    yz_ax.set_xlim([-30, 30])
    yz_ax.set_ylim([-10, 50])
    #ax[0][0].plot(y1, z1)
    # ax[0][0].grid(True)

    xz_ax.set_title("Looking from the side")
    xz_ax.set_xlabel("x")
    xz_ax.set_ylabel("z")
    xz_ax.set_xlim([-30, 30])
    xz_ax.set_ylim([-10, 50])
    #ax[0][1].plot(x1, z1)
    # ax[0][1].grid(True)

    yx_ax.set_title("Looking top down")
    yx_ax.set_xlabel("y")
    yx_ax.set_ylabel("x")
    yx_ax.set_xlim([-30, 30])
    yx_ax.set_ylim([-30, 30])
    yx_ax.invert_yaxis()
    #ax[1][0].plot(y1, -x1)
    # ax[1][0].grid(True)


def lorenz(t, r):
    x, y, z = r
    fx = 10 * (y - x)
    fy = 28 * x - y - x * z
    fz = x * y - (8.0 / 3.0) * z
    return np.array([fx, fy, fz], float)


number = 2
color_list = ["blue", "green", "red"]

t_ev = np.linspace(0, 40, 4000)  # time that's being evaluated
t_sp = [0, len(t_ev)]  # span of the time
shape = (number, len(t_ev))


r0 = [7, 7, 7]
d_0 = [[0, 0, 0],
       [0.01, 0, 0]]  # ,
# [0.001, 0, 0]]


# dZ_0 = np.linalg.norm(d_0) # not accurate for multi dimension yet

r = (np.ones((number, 1)) * r0) + d_0

sol = np.empty(shape, dtype=object)
# for i in range(number):
#	sol[i] = solve_ivp(lorenz, t_span = t_sp, y0 = r[i], t_eval = t_ev)


with open('sol.pkl', 'rb') as f:
    sol = pickle.load(f)


#sol_1 = solve_ivp(lorenz, t_span = t_sp, y0 = r1, t_eval = t_ev)
#print("done solving diffeq")
#t = sol_0.t

#func_1 = sol_0.y
#func_2 = sol_1.y

#x1, y1, z1 = func_1
#x2, y2, z2 = func_2

fig, ax = plt.subplots(2, 2, figsize=(9, 9))

yz_ax = ax[0][0]
xz_ax = ax[0][1]
yx_ax = ax[1][0]

# *r[0][0] Can i just put in the x solution as the data right now?
x_data = np.zeros(shape)
y_data = np.zeros(shape)  # *r0[1] # ab = xy or yz or xz etc.
z_data = np.zeros(shape)  # *r0[2]
#t_data = np.empty(shape)
# dist_data = np.empty(shape)#*np.sqrt(r0[0]**2+r0[1]**2+r0[2]**2)


yz_line = np.empty(shape=(number, 1), dtype=Line2D)
xz_line = np.empty(shape, dtype=Line2D)
yx_line = np.empty(shape, dtype=Line2D)
#diff_line = np.empty(shape, dtype=Line2D)

yz_circ = np.empty(shape, dtype=plt.Circle)
xz_circ = np.empty(shape, dtype=plt.Circle)
yx_circ = np.empty(shape, dtype=plt.Circle)
print(yz_line)
for i in range(number):

    y_data[i] = sol[i][0].y[i][1]
    z_data[i] = sol[i][0].y[i][2]

    yz_circ[i] = plt.Circle(
        (y_data[i][0], z_data[i][0]), .5, color=color_list[i])

    yz_line[i] = Line2D([y_data[i][0]], [z_data[i][0]],
                        linewidth=1., color=color_list[i], zorder=1)
    yz_ax.add_line(yz_line[i][0])
    yz_ax.add_patch(yz_circ[i][0])
    # xz_ax.add_patch(xz_circ[i][0])
    # yx_ax.add_patch(yx_circ[i][0])
# def update(input_data):
# for i in range(number):
#		y_data, z_data

format()
plt.show()
exit()
#xz_circ[i] = plt.Circle((z_data[i][0], x_data[i][0]), .5, color=color_list[i])
#yx_circ[i] = plt.Circle((y_data[i][0], x_data[i][0]), .5, color=color_list[i])
#yz_circ[i][0].center = (y_data[i][0], z_data[i][0])
#xz_circ[i][0].center = (z_data[i][0], x_data[i][0])
#yx_circ[i][0].center = (y_data[i][0], x_data[i][0])

xz_line[i] = Line2D(z_data[i], x_data[i], linewidth=1.,
                    color=color_list[i], zorder=1)
yx_line[i] = Line2D(y_data[i], x_data[i], linewidth=1.,
                    color=color_list[i], zorder=1)
#diff_line[i] = Line2D(t_data[i], dist_data[i], linewidth=0.5, color=color_list[i], zorder=1)


xz_ax.add_line(xz_line[i][0])
yx_ax.add_line(yx_line[i][0])
# ax[1][1].add_line(diff_line[i])


# print(x_data[0][0])
# print(x[1])
#x_data[0][1] = x[1]
# print(x_data[0])

# def update(sol):
for i in range(number):
    xyz = sol[i][0].y
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    x_data[i] = x[0:]
    y_data[i] = y[0:]
    z_data[i] = z[0:]
    #t_data[i] = t_ev[0:]
    #new_dist = np.append(dist_data[i], np.sqrt(x[0:]**2 + y[0:]**2 + z[0:]**2))

    #yz_line[i] = Line2D(y_data[i], z_data[i], linewidth=1., color=color_list[i], zorder=1)
    #xz_line[i] = Line2D(z_data[i], x_data[i], linewidth=1., color=color_list[i], zorder=1)
    yz_line[i] = Line2D(y_data[i], z_data[i], linewidth=1.,
                        color=color_list[i], zorder=1)
    xz_line[i] = Line2D(x_data[i], z_data[i], linewidth=1.,
                        color=color_list[i], zorder=1)
    yx_line[i] = Line2D(y_data[i], x_data[i], linewidth=1.,
                        color=color_list[i], zorder=1)

    #diff_line[i] = Line2D(t_data[i], new_dist, linewidth=0.5, color=color_list[i], zorder=1)

    # yz_ax.add_line(yz_line[i][0])
    # xz_ax.add_line(xz_line[i][0])
    # yx_ax.add_line(yx_line[i][0])
    # ax[1][1].add_line(diff_line[i][0])
    # return yz_line, xz_line, yx_line,
plt.show()
exit()


def update_i(input_data):
    x, y, z, t = input_data

    yz_circ.center = (y, z)
    xz_circ.center = (x, z)
    yx_circ.center = (y, x)

    x_data.append(x)
    y_data.append(y)
    z_data.append(z)
    t_data.append(t)
    dist_data.append(np.sqrt(x**2 + y**2 + z**2))

    yz_line.set_data(y_data[-400:], z_data[-400:])
    xz_line.set_data(x_data[-400:], z_data[-400:])
    yx_line.set_data(y_data[-400:], x_data[-400:])
    diff_line.set_data(t_data, dist_data)

    return yz_line, xz_line, yx_line, diff_line, yz_circ, yx_circ, xz_circ


def update(input_data_list):
    for i in range(number):
        yz_line[i], xz_line[i], yx_line[i], diff_line[i], yz_circ[i], yx_circ[i], xz_circ[i] = update(
            input_data_list[i])


ax[1][1].set_title("Difference")
ax[1][1].set_xlabel("time")
ax[1][1].set_ylabel("distance to origin")
ax[1][1].set_xlim([0, 40])
ax[1][1].set_ylim([0, 50])

yz_ax.set_title("Looking head on")
yz_ax.set_xlabel("y")
yz_ax.set_ylabel("z")
yz_ax.set_xlim([-30, 30])
yz_ax.set_ylim([-10, 50])
#ax[0][0].plot(y1, z1)
# ax[0][0].grid(True)


xz_ax.set_title("Looking from the side")
xz_ax.set_xlabel("x")
xz_ax.set_ylabel("z")
xz_ax.set_xlim([-30, 30])
xz_ax.set_ylim([-10, 50])
#ax[0][1].plot(x1, z1)
# ax[0][1].grid(True)


yx_ax.set_title("Looking top down")
yx_ax.set_xlabel("y")
yx_ax.set_ylabel("x")
yx_ax.set_xlim([-30, 30])
yx_ax.set_ylim([-30, 30])
yx_ax.invert_yaxis()

format()
plt.show()
exit()


plt.tight_layout()

ani = FuncAnimation(fig, update, zip(x1, y1, z1, t), interval=10,
                    blit=True, repeat=False, save_count=len(t_ev))

plt.show()
