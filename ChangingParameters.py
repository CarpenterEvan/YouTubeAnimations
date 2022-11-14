import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')


d_0 = [0.01, 0, 0]
r0 = [0, 0.01, 30]
sigma, rho, beta = (5.5, 28, 8.0 / 3.0)


t_ev = np.linspace(0, 20, 2000)  # time that's being evaluated
t_sp = [0, len(t_ev)]  # span of the time


def get_ranges(x, y, z, classic=True):
    '''Find the maximum and minimum value of the x,y,z solutions, round it to the nearest 5,
    and return (min_rounded, max_rounded) for yz, xz, yx axes'''
    round_to_nearest = 10

    def max_round(q): return (max(q) // round_to_nearest) * \
        round_to_nearest + round_to_nearest
    def min_round(q): return (min(q) // round_to_nearest) * \
        round_to_nearest - round_to_nearest

    def largest_value(q): return max(abs(max_round(q)), abs(min_round(q)))
    axis_limit = max(largest_value(x), largest_value(y), largest_value(z))

    if classic:
        x_ax_limit = np.array((-0.5, 0.5)) * axis_limit
        y_ax_limit = np.array((-0.5, 0.5)) * axis_limit
        z_ax_limit = np.array((0, 1)) * axis_limit
    else:
        x_ax_limit = np.array((-1, 1)) * axis_limit
        y_ax_limit = np.array((-1, 1)) * axis_limit
        z_ax_limit = np.array((-1, 1)) * axis_limit
    return x_ax_limit, y_ax_limit, z_ax_limit


def smallest_and_biggest(x, y, z):
    biggest = max(max(x), max(y), max(z))
    smallest = min(min(x), min(y), min(z))
    return smallest, biggest


def lorenz(t, r, sigma=10, rho=28, beta=8.0 / 3.0):
    x, y, z = r
    fx = sigma * (y - x)
    fy = rho * x - y - x * z
    fz = x * y - beta * z
    return np.array([fx, fy, fz], float)


parameter_space = [sigma]  # np.arange(sigma, sigma, 1)

param_list = np.empty(shape=(len(parameter_space), 3))
solution_list = np.empty(shape=(len(parameter_space), 4), dtype=list)
for i, sigma_i in enumerate(parameter_space):
    full_sol = solve_ivp(lorenz, t_span=t_sp, y0=r0,
                         t_eval=t_ev, args=(sigma_i, rho, beta))
    x, y, z = full_sol.y
    t = full_sol.t
    param_list[i] = (sigma_i, rho, beta)
    solution_list[i] = x, y, z, t
    # maybe a zip?
print(solution_list)
print(param_list)


x_data, y_data, z_data, t_data = solution_list[0][:]
smallest_value, biggest_value = smallest_and_biggest(x_data, y_data, z_data)
x_ax_limit, y_ax_limit, z_ax_limit = get_ranges(
    x_data, y_data, z_data, classic=True)

fig = plt.figure(figsize=(12, 8), layout="tight")
spec = fig.add_gridspec(2, 3)
mytext = matplotlib.text.Text(0.5, 0.5, "XXXXXXXXXX", color="white", zorder=3)
mytext.draw()
plt.show()
exit()
yz_ax = fig.add_subplot(spec[0, 1])
xz_ax = fig.add_subplot(spec[0, 0])
yx_ax = fig.add_subplot(spec[0, 2])
ax4 = fig.add_subplot(spec[1, :])

fig.suptitle(
    r"\noindent$x_0=${:.4f} $y_0=${:.4f} $z_0=${:.4f}\\[0.5cm]$\sigma=${:.4f} $\rho=${:.4f} $\beta=${:.4f}".format(
        r0[0],
        r0[1],
        r0[2],
        sigma,
        rho,
        beta),
    fontsize=14)
yz_ax.set(title="Looking head on",
          xlabel="y", ylabel="z",
          xlim=y_ax_limit, ylim=z_ax_limit)
xz_ax.set(title="Looking from the side",
          xlabel="x", ylabel="z",
          xlim=x_ax_limit, ylim=z_ax_limit)
yx_ax.set(title="Looking top down",
          xlabel="y", ylabel="x",
          xlim=y_ax_limit, ylim=x_ax_limit)
ax4.set(title="Magnitude of each component",
        xlabel="time", ylabel="Magnitude",
        xlim=[t_ev[0], t_ev[-1]], ylim=[smallest_value, biggest_value])

yz_line = Line2D(y_data, z_data, axes=yz_ax, linewidth=1.,
                 color="blue", zorder=1, rasterized=True, animated=False)
xz_line = Line2D(x_data, z_data, axes=xz_ax, linewidth=1.,
                 color="blue", zorder=1, rasterized=True, animated=False)
yx_line = Line2D(y_data, x_data, axes=yx_ax, linewidth=1.,
                 color="blue", zorder=1, rasterized=True, animated=False)

x_line = Line2D(t_data, x_data, linewidth=1., color="cyan",
                zorder=1, label="x", rasterized=True, animated=False)
y_line = Line2D(t_data, y_data, linewidth=1., color="red",
                zorder=1, label="y", rasterized=True, animated=False)
z_line = Line2D(t_data, z_data, linewidth=1., color="green",
                zorder=1, label="z", rasterized=True, animated=False)

yz_ax.add_line(yz_line)
xz_ax.add_line(xz_line)
yx_ax.add_line(yx_line)

ax4.add_line(x_line)
ax4.add_line(y_line)
ax4.add_line(z_line)
yx_ax.invert_yaxis()
ax4.legend()


def update(data):
    param_list, solution_list = data
    sigma, rho, beta = param_list
    x_data, y_data, z_data, t_data = solution_list
    fig.suptitle(
        r"\noindent$x_0=${:.4f} $y_0=${:.4f} $z_0=${:.4f}\\[0.5cm]$\sigma=${:.4f} $\rho=${:.4f} $\beta=${:.4f}".format(
            r0[0],
            r0[1],
            r0[2],
            sigma,
            rho,
            beta),
        fontsize=14)
    fixed_point_a = [np.sqrt(beta * (rho - 1)),
                     np.sqrt(beta * (rho - 1)), rho - 1]
    fixed_point_b = [-np.sqrt(beta * (rho - 1)), -
                     np.sqrt(beta * (rho - 1)), rho - 1]
    fixed_points = np.array([fixed_point_a,
                             fixed_point_b])

    for point in fixed_points:
        x_point, y_point, z_point = point
        yz_ax.add_patch(plt.Circle((y_point, z_point), .05,
                        color="white", rasterized=True))
        xz_ax.add_patch(plt.Circle((x_point, z_point), .05,
                        color="white", rasterized=True))
        yx_ax.add_patch(plt.Circle((y_point, x_point), .05,
                        color="white", rasterized=True))

    unique_fixed_point_magnitudes = [
        -np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1]

    for point in unique_fixed_point_magnitudes:
        ax4_fixed_points_line = ax4.axhline(
            point, linewidth=0.5, color="white", zorder=0, linestyle=":")

    yz_line.set_data(y_data, z_data)
    xz_line.set_data(x_data, z_data)
    yx_line.set_data(y_data, x_data)

    x_line.set_data(t_data, x_data)
    y_line.set_data(t_data, y_data)
    z_line.set_data(t_data, z_data)

    return yz_line, xz_line, yx_line, x_line, y_line, z_line, ax4_fixed_points_line


data = zip(param_list, solution_list)

ani = FuncAnimation(fig, update, data, interval=20, blit=False)

plt.show()
exit()
