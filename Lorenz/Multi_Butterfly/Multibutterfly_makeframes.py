import os
import sys
import pickle
import time
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import multiprocessing 
print(" ")

plt.style.use('dark_background')

number_of_cores_used, dpi_used, fps_used, istest = 2, 100, 20, 1#[int(argument) for argument in sys.argv[1:]]

solution_file_path = f"{sys.path[0]}/Multibutterfly_solution"
print(f"\nOpening solution from {solution_file_path}\n")

solution = pickle.load(open(f"{sys.path[0]}/Multibutterfly_solution", "rb"))


t = solution["time"]
sigma, rho, beta = solution["args"] 
line_segments = np.empty(shape=(2), dtype=tuple)


for index, value in enumerate(solution["sol"]):
    x_i, y_i, z_i = value

    line_segments[index] = np.array([x_i,z_i])
    #line_segments[index] = zip(x_i, z_i)

for i in line_segments:
    print(i)



total_time_sec = t[-1] 
points_per_sec = len(t) / t[-1]

dt = t[-1]/t.size

di = int(1/dt/fps_used)

the_input = np.arange(0,t.size-di, di, dtype=int)

def get_ranges(x,y,z, classic=True):

	'''Find the maximum and minimum value of the x,y,z solutions, round it to the nearest 5, 
	and return (min_rounded, max_rounded) for x, y, z axes. 

    The 'classic' parameters of sigma, rho, beta = (10, 28, 8.0/3.0) yield semi-predictable results
    that I try and optimize the axis scaling for, 
    but the safer option for differing parameters is setting classic to False'''

	round_to_nearest=10

	max_round = lambda q: (max(q) // round_to_nearest) * round_to_nearest + round_to_nearest
	min_round = lambda q: (min(q) // round_to_nearest) * round_to_nearest - round_to_nearest

	largest_value = lambda q: max(abs(max_round(q)), abs(min_round(q)))
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


lines = LineCollection(line_segments, linewidth=1)

class plot_circle(Circle):
	def __init__(self, xy:tuple, fixed_point=False):
		super().__init__(xy)
		if fixed_point:
			self.set_radius(0.15)
			self.set_color("white")
		else: 
			self.set_radius(0.25)
			self.set_color("blue")
		self.set_zorder(2)
		self.set_rasterized(True)


exit()
#################################### Make Artists ####################################
fixed_point_a = [np.sqrt(beta * (rho - 1)), np.sqrt(beta * (rho - 1)), rho-1]
fixed_point_b = [-np.sqrt(beta * (rho - 1)), -np.sqrt(beta * (rho - 1)), rho-1]

yz_circle = plot_circle((y[0], z[0]))
xz_circle = plot_circle((x[0], z[0]))
yx_circle = plot_circle((y[0], x[0]))

fixed_point_1 = plot_circle((fixed_point_a[0], fixed_point_a[2]), fixed_point=True)

fixed_point_2 = plot_circle((fixed_point_b[0], fixed_point_b[2]), fixed_point=True)

xz_line = projection_line([x[0]], [x[0]])



#################################### Make Figure ####################################

fig = plt.figure(figsize=(12, 8), layout="tight")

spec = fig.add_gridspec(1, 1)

ax = fig.add_subplot(spec[0, 0])

x_ax_limit, y_ax_limit, z_ax_limit = get_ranges(x, y, z, classic=True)

####################################### Titles #######################################
initial_condition_string = r"\noindent$x_0=${:.4f} $y_0=${:.4f} $z_0=${:.4f}".format(x[0], y[0], z[0])
parameters_string = r"$\sigma=${:.4f} $\rho=${:.4f} $\beta=${:.4f}".format(sigma, rho, beta)
#fig.suptitle(initial_condition_string + r"\\[0.5cm]" + parameters_string, fontsize=14)
#xz_ax.set(title="Looking from the side",     xlabel="x", ylabel="z")
ax.set_aspect('equal', adjustable='box')
ax.add_patch(xz_circle)
plt.axis('off')


def make_frame(i):
    '''Plots the last 3000 points up to the ith point in the sequence
    and saves the frame as an image.'''
    frame_start = time.time()

    ax.set(xlim = x_ax_limit, ylim = z_ax_limit)

    x_data = x[:i]
    z_data = z[:i]
	#t_data = t[:i]

    ax.add_patch(xz_fixed_point_1)
    ax.add_patch(xz_fixed_point_2)
	
    xz_line.set_data(x_data[-3_000:], z_data[-3_000:])
    ax.add_line(xz_line)

    xz_circle.center = (x[i], z[i])

    plt.savefig(f"{sys.path[0]}/butterfly_frames/_img{i//di:04d}.png", dpi=dpi_used)

    ax.lines.remove(xz_line)
    time_frame_took = f"{time.time()-frame_start:.3f} s"

    if multiprocessing.current_process().name != "MainProcess":
        process_number = int(multiprocessing.current_process().name[-1])
        move = number_of_cores_used+1-process_number

        frame_count = f"{i // di: >{len(str(t.size))}} / {t.size // di}"
        percent_per_core = f"Core {process_number}: {(i // di)/ (t.size//di)*100: >5.2f}%"
		
        #print(f"", end="\r")
        print(f"\033[{move}A\u001b[0K{percent_per_core} -- {time_frame_took}", end="\r")
        print(f"\033[{move}E", end="\r")
    else: 
        print(f"{(i // di)/ (t.size//di)*100: >5.2f}% -- {time_frame_took}", end="\r")


def main():
    os.system(f"cd {sys.path[0]}")
    pool = multiprocessing.Pool(number_of_cores_used)
    old_pngs = glob.glob(f"{sys.path[0]}/butterfly_frames/*.png")
    if len(old_pngs) > 1:
        for png in old_pngs:
            print(png)
            os.remove(png)
    if istest:
        make_frame(the_input[-1])
    else:
        pool.map(make_frame, the_input)

if __name__ == "__main__":
    main()