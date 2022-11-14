import multiprocessing
import time
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
plt.style.use('dark_background')


class axis_line(Line2D):
    def __init__(self, xdata: list, ydata: list):
        super().__init__(xdata, ydata)
        self.set_linewidth(0.75)
        self.set_color("blue")
        self.set_zorder(1)
        self.set_rasterized(True)


plt.xkcd()
fig = plt.figure()
x = np.linspace(0, 4, 50)
y = np.sin(x)
plt.plot(x, y)
plt.show()
