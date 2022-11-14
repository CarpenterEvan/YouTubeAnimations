import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')
from matplotlib.lines import Line2D
import time
import multiprocessing
import os

start = time.time()
fps = 50
total_time = 5
points_per_sec = 100
nsteps = total_time * points_per_sec
dt = total_time/nsteps
d_0 = [0.01, 0, 0]
r0 = [0, 1, 30] 
sigma, rho, beta = (10, 28, 8.0/3.0)
number_of_cores_used = 6
di = int(1/fps/dt)
t_ev = np.linspace(0,total_time,nsteps) # time that's being evaluated
t_sp = [0, len(t_ev)] # span of the time


def get_ranges(x,y,z, classic=True):
	'''Find the maximum and minimum value of the x,y,z solutions, round it to the nearest 5, 
	and return (min_rounded, max_rounded) for yz, xz, yx axes'''
	round_to_nearest=10
	max_round = lambda q: (max(q) // round_to_nearest) * round_to_nearest + round_to_nearest
	min_round = lambda q: (min(q) // round_to_nearest) * round_to_nearest - round_to_nearest
	largest_value = lambda q: max(abs(max_round(q)), abs(min_round(q)))
	axis_limit = max(largest_value(x), largest_value(y), largest_value(z))

	if classic:
		x_ax_limit = np.array((-0.5, 0.5))*axis_limit
		y_ax_limit = np.array((-0.5, 0.5))*axis_limit
		z_ax_limit = np.array((0, 1))*axis_limit
	else:
		x_ax_limit = np.array((-1, 1))*axis_limit
		y_ax_limit = np.array((-1, 1))*axis_limit
		z_ax_limit = np.array((-1, 1))*axis_limit
	return x_ax_limit, y_ax_limit, z_ax_limit

def smallest_and_biggest(x,y,z):
	biggest = max(max(x), max(y), max(z))
	smallest = min(min(x), min(y), min(z))
	return smallest, biggest

def lorenz(t, r, sigma=10, rho=28, beta=8.0/3.0):
    x, y, z = r
    fx = sigma * (y - x)
    fy = rho * x - y - x * z
    fz =  x * y - beta * z
    return np.array([fx, fy, fz], float)


def solving():
	print("about to solve")
	sol = solve_ivp(lorenz, t_span = t_sp, y0 = r0, t_eval = t_ev, args=(sigma, rho, beta))
	print("Done Solving")
	return sol

class projection_line(Line2D):
	def __init__(self, xdata:list, ydata:list):
		super().__init__(xdata, ydata)
		self.set_linewidth(0.6)
		self.set_color("blue")
		self.set_zorder(1)
		self.set_rasterized(True)
class plot_circle(Circle):
	def __init__(self, xy:tuple, fixed_point=False):
		super().__init__(xy)
		if fixed_point:
			self.set_radius(0.25)
			self.set_color("white")
		else: 
			self.set_radius(0.5)
			self.set_color("blue")
		self.set_zorder(2)
		self.set_rasterized(True)

sol = solving()
t = sol.t
x, y, z = sol.y

#################################### Make Artists ####################################
fixed_point_a = [np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho-1]
fixed_point_b = [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho-1]

yz_circle = plot_circle((r0[1], r0[2]))
xz_circle = plot_circle((r0[0], r0[2]))
yx_circle = plot_circle((r0[1], r0[0]))

yz_fixed_point_1 = plot_circle((fixed_point_a[1], fixed_point_a[2]), fixed_point=True)
xz_fixed_point_1 = plot_circle((fixed_point_a[0], fixed_point_a[2]), fixed_point=True)
yx_fixed_point_1 = plot_circle((fixed_point_a[1], fixed_point_a[1]), fixed_point=True)

yz_fixed_point_2 = plot_circle((fixed_point_b[1], fixed_point_b[2]), fixed_point=True)
xz_fixed_point_2 = plot_circle((fixed_point_b[0], fixed_point_b[2]), fixed_point=True)
yx_fixed_point_2 = plot_circle((fixed_point_b[1], fixed_point_b[1]), fixed_point=True)


yz_line = projection_line([r0[1]], [r0[2]])
xz_line = projection_line([r0[0]], [r0[2]])
yx_line = projection_line([r0[1]], [r0[0]])

x_line = Line2D([t_ev[0]], [r0[1]], linewidth=1., color="red", zorder=1, label="x", rasterized=True)
y_line = Line2D([t_ev[0]], [r0[2]], linewidth=1., color="green", zorder=1, label="y", rasterized=True)
z_line = Line2D([t_ev[0]], [r0[2]], linewidth=1., color="cyan", zorder=1, label="z", rasterized=True)


#################################### Make Figure ####################################
fig = plt.figure(figsize=(12, 8), layout="tight")

spec = fig.add_gridspec(2, 3)

xz_ax = fig.add_subplot(spec[0, 0])
yz_ax = fig.add_subplot(spec[0, 1])
yx_ax = fig.add_subplot(spec[0, 2])
ax4 = fig.add_subplot(spec[1, :])

	

	

smallest_value, biggest_value = smallest_and_biggest(x, y, z)
x_ax_limit, y_ax_limit, z_ax_limit = get_ranges(x, y, z, classic=True)

initial_condition_string = r"\noindent$x_0=${:.4f} $y_0=${:.4f} $z_0=${:.4f}".format(r0[0], r0[1], r0[2])
parameters_string = r"$\sigma=${:.4f} $\rho=${:.4f} $\beta=${:.4f}".format(sigma, rho, beta)
fig.suptitle(initial_condition_string + r"\\[0.5cm]" + parameters_string, fontsize=14)
yz_ax.set(title="Looking head on",           xlabel="y", ylabel="z")
xz_ax.set(title="Looking from the side",     xlabel="x", ylabel="z")
yx_ax.set(title="Looking top down",          xlabel="y", ylabel="x")
ax4.set(title="Magnitude of each component", xlabel="time", ylabel="Magnitude")

ax4.axhline(-np.sqrt(beta * (rho - 1)), linewidth=0.5, color="white", zorder=0, linestyle=":") # plot horizontal dotted line for each fixed point
ax4.axhline(np.sqrt(beta * (rho - 1)), linewidth=0.5, color="white", zorder=0, linestyle=":")
ax4.axhline(rho - 1, linewidth=0.5, color="white", zorder=0, linestyle=":")


def make_frame(i):
	
	frame_start = time.time()

	yz_ax.set(xlim = y_ax_limit, ylim = z_ax_limit)
	xz_ax.set(xlim = x_ax_limit, ylim = z_ax_limit)
	yx_ax.set(xlim = y_ax_limit, ylim = x_ax_limit)

	ax4.set(xlim=[t_ev[0],t_ev[-1]], ylim=[smallest_value, biggest_value])
	
	x_data = x[:i]
	y_data = y[:i]
	z_data = z[:i]
	t_data = t[:i]


	yz_ax.add_patch(yz_fixed_point_1)
	xz_ax.add_patch(xz_fixed_point_1)
	yx_ax.add_patch(yx_fixed_point_1)

	yz_ax.add_patch(yz_fixed_point_2)
	xz_ax.add_patch(xz_fixed_point_2)
	yx_ax.add_patch(yx_fixed_point_2)

	yz_ax.add_patch(yz_circle)
	xz_ax.add_patch(xz_circle)
	yx_ax.add_patch(yx_circle)

	yz_line.set_data(y_data, z_data)
	xz_line.set_data(x_data, z_data)
	yx_line.set_data(y_data, x_data)

	x_line.set_data(t_data, x_data)
	y_line.set_data(t_data, y_data)
	z_line.set_data(t_data, z_data)
	

	yz_ax.add_line(yz_line)
	xz_ax.add_line(xz_line)
	yx_ax.add_line(yx_line)


	ax4.add_line(x_line)
	ax4.add_line(y_line)
	ax4.add_line(z_line)

	yz_circle.center = (y[i], z[i])
	xz_circle.center = (x[i], z[i])
	yx_circle.center = (y[i], x[i])
	
	plt.savefig(f"butterfly_frames/_img{i//di:04d}.png", dpi=200)
	yz_ax.lines.remove(yz_line)
	xz_ax.lines.remove(xz_line)
	yx_ax.lines.remove(yx_line)
	ax4.lines.remove(x_line)
	ax4.lines.remove(y_line)
	ax4.lines.remove(z_line)
	process_number = int(multiprocessing.current_process().name[-1])
	move = number_of_cores_used+1-process_number
	print(f"\033[{move}A", end="\r")
	print(f"{i // di: >{len(str(t_ev.size))}} / {t_ev.size // di} -- Core {process_number}: {(i // di)/ (t.size//di)*100:.2f}% -- {time.time()-frame_start:.3f}", end="\r")
	print(f"\033[{move}E", end="\r")

def main():
	
	pool = multiprocessing.Pool(number_of_cores_used)
	print(f"{di=}")
	the_input = range(0,t_ev.size, di)
	pool.map(make_frame, the_input)
	

if __name__ == "__main__":
	main()