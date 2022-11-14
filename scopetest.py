from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import numpy as np
import sys
from matplotlib import pyplot as plt
plt.style.use('dark_background')

d_0 = [0.000000001, 0, 0]
r0 = [0, 2, 0]
r1 = [r0[i] + d_0[i] for i in range(3)]
dZ_0 = np.linalg.norm(d_0)
t_ev = np.linspace(0, 40, 4000)  # time that's being evaluated

t_sp = [0, len(t_ev)]  # span of the time


def lorenz(t, r):
    x, y, z = r
    fx = 10 * (y - x)
    fy = 28 * x - y - x * z
    fz = x * y - (8.0 / 3.0) * z
    return np.array([fx, fy, fz], float)


sol_0 = solve_ivp(lorenz, t_span=t_sp, y0=r0, t_eval=t_ev)
sol_1 = solve_ivp(lorenz, t_span=t_sp, y0=r1, t_eval=t_ev)
print("done solving diffeq")
t = sol_0.t
y0_x, y0_y, y0_z = sol_0.y

y1_x, y1_y, y1_z = sol_1.y


class Scope:
    def __init__(self, ax, dt):

        self.ax0 = ax[0]
        self.ax1 = ax[1]
        self.dt = dt
        self.tdata = [0]
        self.y1_data = [0]
        self.y2_data = [0]
        self.y_diff = [0]

        self.line1 = Line2D(self.tdata, self.y1_data,
                            linewidth=1., color="blue", zorder=1)
        self.line2 = Line2D(self.tdata, self.y2_data,
                            linewidth=1., color="red", zorder=0)
        self.line3 = Line2D(self.tdata, self.y_diff,
                            linewidth=1., color="purple")
        self.ax0.add_line(self.line1)
        self.ax0.add_line(self.line2)
        self.ax1.add_line(self.line3)

    def update(self, y):

        y1, y2 = y

        t = self.tdata[-1] + self.dt

        self.tdata.append(t)
        self.y1_data.append(y1)
        self.y2_data.append(y2)
        self.y_diff.append(np.abs(y1-y2))

        self.line1.set_data(self.tdata, self.y1_data)
        self.line2.set_data(self.tdata, self.y2_data)
        self.line3.set_data(self.tdata, self.y_diff)

        return self.line1, self.line2, self.line3


fig, ax = plt.subplots(2, 1, figsize=(16, 9))

maximum_value_rounded = (max(max(y0_x), max(y1_x)) // 5) * 5 + 5
ax[0].set_xlim([t_ev[0], t_ev[-1]])
ax[0].set_ylim([-maximum_value_rounded, maximum_value_rounded])


maximum_difference_rounded = (max(y0_x-y1_x) // 5) * 5 + 5
ax[1].set_xlim([t_ev[0], t_ev[-1]])
ax[1].set_ylim([-0.5, maximum_difference_rounded])

ax[0].set_title(
    f"Initial Condition of Blue line: {r0}" + "\n" + f"Initial Condition of Red line: {r1}")
ax[0].set_xlabel("Time")
ax[1].set_xlabel("Time")
ax[0].set_ylabel("x-component of vectors")
ax[0].set_xlabel("time")
ax[1].set_ylabel("Magnitude")
ax[1].set_title("Absolute Difference")
plt.tight_layout()
if sys.argv[-1] == "image":
    ax[0].plot(t, y0_x)
    ax[0].plot(t, y1_x)
    ax[1].plot(t, abs(y0_x-y1_x))
    plt.show()
    exit()

scope = Scope(ax, dt=t_ev[-1]/len(t_ev))
print("starting animation")
ani = FuncAnimation(fig, scope.update, zip(y0_x, y1_x),
                    interval=30, blit=True, repeat=False, save_count=len(t_ev))
plt.show()
if input("Save? [y/n]: ") == "y":
    print("saving animation")
    writervideo = FFMpegWriter(fps=60, bitrate=-1)
    ani.save('chaos.mp4', writer=writervideo, dpi=120)
    print("done!")
