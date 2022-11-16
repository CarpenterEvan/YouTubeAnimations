import os
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')
fps = 10


d_0 = [0.01, 0, 0]
r0 = [0, 1, 30]
sigma, rho, beta = (6, 28, 8.0 / 3.0)

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


sol = solve_ivp(
    lorenz,
    t_span=t_sp,
    y0=r0,
    t_eval=t_ev,
    args=(
        sigma,
        rho,
        beta))
t = sol.t
x, y, z = sol.y
smallest_value, biggest_value = smallest_and_biggest(x, y, z)
x_ax_limit, y_ax_limit, z_ax_limit = get_ranges(x, y, z, classic=True)

fig = plt.figure(figsize=(12, 8), layout="tight")
spec = fig.add_gridspec(2, 3)

xz_ax = fig.add_subplot(spec[0, 0])
yz_ax = fig.add_subplot(spec[0, 1])
yx_ax = fig.add_subplot(spec[0, 2])
ax4 = fig.add_subplot(spec[1, :])

unique_fixed_point_magnitudes = [-np.sqrt(beta * (rho - 1)),
                                 np.sqrt(beta * (rho - 1)), rho - 1]
for point in unique_fixed_point_magnitudes:
    ax4.axhline(point, linewidth=0.5, color="white", zorder=0, linestyle=":")

fig.suptitle(
    r"\noindent$x_0=${:.4f} $y_0=${:.4f} $z_0=${:.4f}\\[0.5cm]$\sigma=${:.4f} $\rho=${:.4f} $\beta=${:.4f}".format(
        r0[0],
        r0[1],
        r0[2],
        sigma,
        rho,
        beta),
    fontsize=14)

xz_ax.set(title="Looking from the side",
          xlabel="x", ylabel="z",
          xlim=x_ax_limit, ylim=z_ax_limit)

yx_ax.set(title="Looking top down",
          xlabel="y", ylabel="x",
          xlim=y_ax_limit, ylim=x_ax_limit)

ax4.set(title="Magnitude of each component",
        xlabel="time", ylabel="Magnitude",
        xlim=[t_ev[0], t_ev[-1]], ylim=[smallest_value, biggest_value])


#yz_ax = ax[0][0]
#xz_ax = ax[0][1]
#yx_ax = ax[1][0]
#ax4 = ax[1][1]

x_data = [x]
y_data = [y]  # ab = xy or yz or xz etc.
z_data = [z]
t_data = [t]

yz_line = Line2D(
    y_data,
    z_data,
    linewidth=1.,
    color="blue",
    zorder=1,
    rasterized=True,
    animated=True)
xz_line = Line2D(
    z_data,
    x_data,
    linewidth=1.,
    color="blue",
    zorder=1,
    rasterized=True,
    animated=True)
yx_line = Line2D(
    y_data,
    x_data,
    linewidth=1.,
    color="blue",
    zorder=1,
    rasterized=True,
    animated=True)

x_line = Line2D(
    t_data,
    x_data,
    linewidth=1.,
    color="cyan",
    zorder=1,
    label="x",
    rasterized=True,
    animated=True)
y_line = Line2D(
    t_data,
    y_data,
    linewidth=1.,
    color="red",
    zorder=1,
    label="y",
    rasterized=True,
    animated=True)
z_line = Line2D(
    t_data,
    z_data,
    linewidth=1.,
    color="green",
    zorder=1,
    label="z",
    rasterized=True,
    animated=True)


fixed_point_a = [np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho - 1]
fixed_point_b = [-np.sqrt(beta * (rho - 1)), -
                 np.sqrt(beta * (rho - 1)), rho - 1]
fixed_points = np.array([fixed_point_a,
                         fixed_point_b])

yz_circ = plt.Circle((0, 0), .5, color="blue")
xz_circ = plt.Circle((0, 0), .5, color="blue")
yx_circ = plt.Circle((0, 0), .5, color="blue")
yz_ax.set(title="Looking head on",
          xlabel="y", ylabel="z")
dt = 20 / 2000
fps = 10
print(f"{1/fps/dt=}")
di = int(1 / fps / dt)
print(f"{di=}")


def make_frame(i):
    yz_ax.set(xlim=y_ax_limit, ylim=z_ax_limit)

    y_data = y[:i]
    z_data = z[:i]
    yz_line = Line2D(
        y_data,
        z_data,
        linewidth=1.,
        color="blue",
        zorder=1,
        rasterized=True,
        animated=True)
    yz_ax.add_line(yz_line)

    # xz_ax.add_line(xz_line[:i])
    # yx_ax.add_line(yx_line[:i])

    # ax4.add_line(x_line[:i])
    # ax4.add_line(y_line[:i])
    # ax4.add_line(z_line[:i])

    # for point in fixed_points:
    #	x_point, y_point, z_point = point
    #	yz_ax.add_patch(plt.Circle((y_point, z_point), .05, color="white", rasterized=True))
    #	xz_ax.add_patch(plt.Circle((x_point, z_point), .05, color="white", rasterized=True))
    #	yx_ax.add_patch(plt.Circle((y_point, x_point), .05, color="white", rasterized=True))

    # yz_ax.add_patch(yz_circ)
    # xz_ax.add_patch(xz_circ)
    # yx_ax.add_patch(yx_circ)

    plt.savefig(
        f'ChaosAnimations/butterfly_frames/_img{i//di:04d}.png',
        dpi=100)
    yz_ax.cla()


for i in range(0, t.size, di):
    #print(f" {(i // di)/ (t.size//di)*100:.2f}%")
    print(f"{i // di} / {t.size // di} -- {(i // di)/ (t.size//di)*100:.2f}%")
    #print("\033[1A", end="")
    make_frame(i)

os.system(f"ffmpeg -r {fps} -f image2 -s 576x432 -i ChaosAnimations/butterfly_frames/_img%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p butterfly_test.mp4")
exit()


def update(input_data):
    x, y, z, t = input_data

    yz_circ.center = (y, z)
    xz_circ.center = (x, z)
    yx_circ.center = (y, x)

    x_data.append(x)
    y_data.append(y)
    z_data.append(z)
    t_data.append(t)

    yz_line.set_data(y_data[-400:], z_data[-400:])
    xz_line.set_data(x_data[-400:], z_data[-400:])
    yx_line.set_data(y_data[-400:], x_data[-400:])

    x_line.set_data(t_data, x_data)
    y_line.set_data(t_data, y_data)
    z_line.set_data(t_data, z_data)

    return yz_line, xz_line, yx_line, x_line, y_line, z_line, yz_circ, yx_circ, xz_circ


plt.show()
exit()


ani = FuncAnimation(fig, update, zip(x, y, z, t), interval=20,
                    blit=True, repeat=False, save_count=len(t_ev))


save = "y"  # input("Save? [y/n]: ")
name_of_file = f"chaos_s{sigma:.0f}b{beta:.0f}r{rho:.0f}_x{r0[0]:.0f}y{r0[1]:.0f}z{r0[2]:.0f}.mp4"
audio_file = "ILYBaby.mp3"

# exit()
if save == "n":
    print("saving animation")

    writervideo = FFMpegWriter(
        fps=60,
        bitrate=-1,
        codec="libx264",
        extra_args=[
            '-pix_fmt',
            'yuv420p'])

    ani.save(name_of_file, writer=writervideo, dpi=120)
    print("done!")
else:
    #os.system("afplay " + audio_file)
    plt.show()

print(
    f"ffmpeg -i {name_of_file} -i {audio_file} -c:v copy -c:a aac -shortest happy_birthday_erin.mp4")
