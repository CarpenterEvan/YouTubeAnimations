import pickle
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import os
import glob
import multiprocessing; print(" ")
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')
solution_file_path = f"{sys.path[0]}/butterfly_solution"

print(f"\nOpening solution from {solution_file_path}\n")
solution = pickle.load(open(f"{sys.path[0]}/butterfly_solution", "rb"))
x, y, z = solution["sol"].y
t = solution["sol"].t
sigma, rho, beta = solution["args"] 

number_of_cores_used = 2
fps = 24
total_time_sec = t[-1] 
points_per_sec = len(t) / t[-1]

dt = total_time_sec/t.size
di = int(1/dt/fps)

print(f"{total_time_sec=}")
print(f"{points_per_sec=}")
print(f"{fps=}")
print(f"{dt=}")
print(f"{di=}")

the_input = np.linspace(0,t.size-di, fps, dtype=int, endpoint=True)

print(f"{the_input=}")
exit()

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


#################################### Make Artists ####################################
fixed_point_a = [np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho-1]
fixed_point_b = [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho-1]

yz_circle = plot_circle((y[0], z[0]))
xz_circle = plot_circle((x[0], z[0]))
yx_circle = plot_circle((y[0], x[0]))

xz_fixed_point_1 = plot_circle((fixed_point_a[0], fixed_point_a[2]), fixed_point=True)

xz_fixed_point_2 = plot_circle((fixed_point_b[0], fixed_point_b[2]), fixed_point=True)

xz_line = projection_line([x[0]], [x[0]])

x_line = Line2D([t[0]], [x[0]], linewidth=1., color="red", zorder=1, label="x", rasterized=True)
y_line = Line2D([t[0]], [y[0]], linewidth=1., color="green", zorder=1, label="y", rasterized=True)
z_line = Line2D([t[0]], [z[0]], linewidth=1., color="cyan", zorder=1, label="z", rasterized=True)



#################################### Make Figure ####################################
fig = plt.figure(figsize=(12, 8))#, layout="tight")

spec = fig.add_gridspec(1, 1)

xz_ax = fig.add_subplot(spec[0, 0])

smallest_value, biggest_value = smallest_and_biggest(x, y, z)
x_ax_limit, y_ax_limit, z_ax_limit = get_ranges(x, y, z, classic=True)

####################################### Titles #######################################
initial_condition_string = r"\noindent$x_0=${:.4f} $y_0=${:.4f} $z_0=${:.4f}".format(x[0], y[0], z[0])
parameters_string = r"$\sigma=${:.4f} $\rho=${:.4f} $\beta=${:.4f}".format(sigma, rho, beta)
#fig.suptitle(initial_condition_string + r"\\[0.5cm]" + parameters_string, fontsize=14)
#xz_ax.set(title="Looking from the side",     xlabel="x", ylabel="z")
xz_ax.set_aspect('equal', adjustable='box')
plt.axis('off')
xz_ax.add_patch(xz_circle)



def make_frame(i):

	
	frame_start = time.time()

	xz_ax.set(xlim = x_ax_limit, ylim = z_ax_limit)

	x_data = x[:i]
	z_data = z[:i]
	#t_data = t[:i]

	xz_ax.add_patch(xz_fixed_point_1)
	xz_ax.add_patch(xz_fixed_point_2)
	
	xz_line.set_data(x_data, z_data)
	xz_ax.add_line(xz_line)

	xz_circle.center = (x[i], z[i])
	
	plt.savefig(f"{sys.path[0]}/butterfly_frames/_img{int(t[i]//di):04d}.png", dpi=20)

	xz_ax.lines.remove(xz_line)

	process_number = int(multiprocessing.current_process().name[-1])
	move = number_of_cores_used+1-process_number

	frame_count = f"{i // di: >{len(str(t.size))}} / {t.size // di}"
	percent_per_core = f"Core {process_number}: {(i // di)/ (t.size//di)*100: >5.2f}%"
	time_frame_took = f"{time.time()-frame_start:.3f} s"

	#print(f"", end="\r")
	print(f"\033[{move}A\u001b[0K{percent_per_core} -- {time_frame_took}", end="\r")
	print(f"\033[{move}E", end="\r")


def main():
	os.system(f"cd {sys.path[0]}; ls")
	pool = multiprocessing.Pool(number_of_cores_used)
	#old_pngs = glob.glob(f"butterfly_frames/*.png")
	#if len(old_pngs) > 1:
	#	for png in old_pngs:
	#		os.remove(png)
	print(f"{the_input=}")
	pool.map(make_frame, the_input)

video_parameters_file = open(f"{sys.path[0]}/video_parameters", "wb")
video_parameters = {"fps": fps, 
					"frames_folder": "butterfly_frames",
					"file_name" : "butterfly_test"}
pickle.dump(video_parameters, video_parameters_file, )
video_parameters_file.close()

if __name__ == "__main__":
	
	main()