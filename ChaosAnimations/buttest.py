import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d
import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
from celluloid
import Camera

plt.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection = '3d')
camera = Camera(fig)
i = 10
x = []
y = []
z = []
# Prepare arrays x, y, z
for theta in np.arange(0, 4 * np.pi, 0.05):
   z += [i]
r = i ** 2 + 1
x += [r * math.sin(theta)]
y += [r * math.cos(theta)]
i += 1
ax.plot(x, y, z, color = 'blue')
camera.snap()
anim = camera.animate(blit = False, interval = 10)
anim.save('3d.mp4')