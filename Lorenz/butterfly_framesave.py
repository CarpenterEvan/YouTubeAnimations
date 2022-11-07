import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')
from matplotlib.lines import Line2D

import time

start = time.time()
fps = 60
total_time = 20
nsteps = 2000
dt = total_time/nsteps
d_0 = [0.01, 0, 0]
r0 = [0, 1, 30] 
sigma, rho, beta = (6, 28, 8.0/3.0)


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


sol = solve_ivp(lorenz, t_span = t_sp, y0 = r0, t_eval = t_ev, args=(sigma, rho, beta))
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

fixed_point_a = [np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho-1]
fixed_point_b = [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho-1]
fixed_points = np.array([fixed_point_a,
						fixed_point_b])

yz_circ = plt.Circle((0, 0), .5, color="blue")
xz_circ = plt.Circle((0, 0), .5, color="blue")
yx_circ = plt.Circle((0, 0), .5, color="blue")


fig.suptitle(r"\noindent$x_0=${:.4f} $y_0=${:.4f} $z_0=${:.4f}\\[0.5cm]$\sigma=${:.4f} $\rho=${:.4f} $\beta=${:.4f}".format(r0[0], r0[1], r0[2], sigma, rho, beta), fontsize=14)
yz_ax.set(title="Looking head on",           xlabel="y", ylabel="z")
xz_ax.set(title="Looking from the side",     xlabel="x", ylabel="z")
yx_ax.set(title="Looking top down",          xlabel="y", ylabel="x")
ax4.set(title="Magnitude of each component", xlabel="time", ylabel="Magnitude")

di = int(1/fps/dt)

yz_line = Line2D([y[0]], [z[0]], linewidth=1., color="blue", zorder=1, rasterized=True)
xz_line = Line2D([x[0]], [z[0]], linewidth=1., color="blue", zorder=1, rasterized=True)
yx_line = Line2D([y[0]], [x[0]], linewidth=1., color="blue", zorder=1, rasterized=True)

x_line = Line2D([t[0]], [x[0]], linewidth=1., color="red", zorder=1, label="x", rasterized=True)
y_line = Line2D([t[0]], [y[0]], linewidth=1., color="green", zorder=1, label="x", rasterized=True)
z_line = Line2D([t[0]], [z[0]], linewidth=1., color="cyan", zorder=1, label="x", rasterized=True)

ax4.axhline(-np.sqrt(beta * (rho - 1)), linewidth=0.5, color="white", zorder=0, linestyle=":")
ax4.axhline(np.sqrt(beta * (rho - 1)), linewidth=0.5, color="white", zorder=0, linestyle=":")
ax4.axhline(rho - 1, linewidth=0.5, color="white", zorder=0, linestyle=":")

def make_frame(i):

	frame_start = time.time()
	

	yz_ax.set(xlim=y_ax_limit, ylim=z_ax_limit)
	xz_ax.set(xlim=x_ax_limit, ylim=z_ax_limit)
	yx_ax.set(xlim=y_ax_limit, ylim=x_ax_limit)
	ax4.set(xlim=[t_ev[0],t_ev[-1]], ylim=[smallest_value, biggest_value])
	x_data = x[:i]
	y_data = y[:i]
	z_data = z[:i]
	t_data = t[:i]

	yz_line.set_data(y_data, z_data)
	xz_line.set_data(x_data, z_data)
	yx_line.set_data(y_data, x_data)

	x_line.set_data(t_data, x_data)
	y_line.set_data(t_data, y_data)
	z_line.set_data(t_data, z_data)
	

	yz_ax.add_line(yz_line)
	xz_ax.add_line(xz_line)
	yx_ax.add_line(yx_line)

	

	#xz_ax.add_line(xz_line[:i])
	#yx_ax.add_line(yx_line[:i])

	ax4.add_line(x_line)
	ax4.add_line(y_line)
	ax4.add_line(z_line)

	#for point in fixed_points:
	#	x_point, y_point, z_point = point
	#	yz_ax.add_patch(plt.Circle((y_point, z_point), .05, color="white", rasterized=True))
	#	xz_ax.add_patch(plt.Circle((x_point, z_point), .05, color="white", rasterized=True))
	#	yx_ax.add_patch(plt.Circle((y_point, x_point), .05, color="white", rasterized=True))

	#yz_ax.add_patch(yz_circ)
	#xz_ax.add_patch(xz_circ)
	#yx_ax.add_patch(yx_circ)
	
	plt.savefig(f'butterfly_frames/_img{i//di:04d}.png', dpi=200)
	yz_ax.lines.remove(yz_line)
	xz_ax.lines.remove(xz_line)
	yx_ax.lines.remove(yx_line)
	ax4.lines.remove(x_line)
	ax4.lines.remove(y_line)
	ax4.lines.remove(z_line)
	print(f" {i // di} / {t.size // di} -- {(i // di)/ (t.size//di)*100:.2f}% -- {time.time()-frame_start:.3f}", end="\r")

import multiprocessing

def main():
	pool = multiprocessing.Pool(4)
	print(f"{di=}")
	the_input = range(0,t.size, di)
	pool.map(make_frame, the_input)
if __name__ == "__main__":
	main()
