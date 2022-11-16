import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
plt.rcParams['text.usetex'] = True
plt.style.use('dark_background')

L1, L2 = 1, 1
m1, m2 = 1, 1
g = 9.81


def deriv(t, y, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1dot = z1
    z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1**2 * c +
             L2 * z2**2) - (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s**2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (L1 * z1**2 * s - g * np.sin(theta2) + g *
             np.sin(theta1) * c) + m2 * L2 * z2**2 * s * c) / L2 / (m1 + m2 * s**2)
    return theta1dot, z1dot, theta2dot, z2dot


def calc_E(y_0):
    """Return the total energy of the system."""
    th1, th1d, th2, th2d = y_0
    V = -(m1 + m2) * L1 * g * np.cos(th1) - m2 * L2 * g * np.cos(th2)
    T = 0.5 * m1 * (L1 * th1d)**2 + 0.5 * m2 * ((L1 * th1d)**2 +
                                                (L2 * th2d)**2 + 2 * L1 * L2 * th1d * th2d * np.cos(th1 - th2))
    return T + V


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 10, 0.01
t = np.arange(0, tmax + dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
y_0 = np.array([np.pi,  # theta1
               0,         # z1
               0.1,  # theta2
               0          # z2
                ])


y = solve_ivp(deriv,
              t_span=(t[0],
                      t[-1]),
              y0=y_0,
              t_eval=t,
              args=(L1,
                    L2,
                    m1,
                    m2),
              method="RK45",
              atol=0.0000001,
              rtol=0.00000001)
# Check that the calculation conserves total energy to within some tolerance.
EDRIFT = 0.01
# Total energy from the initial conditions
E = calc_E(y_0)
print(E)
print(calc_E(y.y))
plt.plot(t, np.abs(calc_E(y.y) - E))
plt.show()

if np.max(np.sum(np.abs(calc_E(y.y) - E))) > EDRIFT:
    sys.exit(f'Maximum energy drift of {EDRIFT} exceeded.')
else:
    pass

# Unpack z and theta as a function of time
theta1, theta2 = y.y[0], y.y[2]

# Convert to Cartesian coordinates of the two bob positions.
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

dist = np.sqrt(y2**2 + x2**2)
filtered_index = []
filtered_points = []

# Plotted bob circle radius
r = 0.05


def make_frame(i, bang=False):
    # Plot and save an image of the double pendulum configuration for time
    # point i.
    # The pendulum rods.
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, color='b')
    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    #c0 = Circle((0, 0), 2, edgecolor="cyan", linewidth=0.5, fill=False, zorder=10)
    c1 = Circle((x1[i], y1[i]), r, fc='r', ec='r', zorder=10)
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
    # ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    if bang:
        ax.text(1, 1, "Bang!")
    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-L1 - L2 - r, L1 + L2 + r)
    ax.set_ylim(-L1 - L2 - r, L1 + L2 + r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(f'frames/_img{i//di:04d}.png', dpi=200)
    plt.cla()


# Make an image every di time points, corresponding to a frame rate of fps
# frames per second.
# Frame rate, s-1
fps = 60
di = int(1 / fps / dt)
print(f"{di=}")

fig = plt.figure(figsize=(8, 6), dpi=200)
ax = fig.add_subplot(111)

for index, value in enumerate(dist):
    if index % 2 == 0:
        if value >= (L1 + L2 - 0.01):
            filtered_index.append(index)
            filtered_points.append(value)
        else:
            pass
    else:
        pass
#ax.plot(t, dist)
#ax.scatter(filtered_index, filtered_points)
#ax.axhline( L1+L2-0.001)
# plt.show()

for i in range(0, t.size, di):
    print(f" {(i // di)/ (t.size//di)*100:.2f}%")
    print("\033[1A", end="")
    if i in filtered_index:
        bang = True
    else:
        bang = False
    make_frame(i, bang=bang)

os.system(
    f"ffmpeg -r {fps} -f image2 -s 576x432 -i frames/_img%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test3.mp4")
