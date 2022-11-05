import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

d_0 = [0.001, 0, 0]
r0 = [10, 5, 4] 
r1 = [r0[i] + d_0[i] for i in range(3)] 
r2 = [r0[i] + 0.1*d_0[i] for i in range(3)] 

t_ev = np.linspace(0,40,1000)
t_sp = [0, len(t_ev)]

def lorenz(t, r):
    x, y, z = r
    fx = 10 * (y - x)
    fy = 28 * x - y - x * z
    fz =  x * y - (8.0 / 3.0) * z
    return np.array([fx, fy, fz], float)

sol_0 = solve_ivp(lorenz, t_span = t_sp, y0 = r0, t_eval = t_ev)
sol_1 = solve_ivp(lorenz, t_span = t_sp, y0 = r1, t_eval = t_ev)
sol_2 = solve_ivp(lorenz, t_span = t_sp, y0 = r2, t_eval = t_ev)

t = sol_0.t
y0_x, y0_y, y0_z = sol_0.y
y1_x, y1_y, y1_z = sol_1.y

fig, ax = plt.subplots(2,1)
ax[0].set_title("Initial condition 1: {}\n Initial condition 2: {}".format(r0,r1))
ax[1].set_ylim([-1, 30])
ax[0].set_xlim([0, 40])
ax[0].set_ylim([-20, 20])
ax[1].set_xlim([0, 40])
ax[1].set_title("Absolute Difference")

tdata = []
y0data = []
y1data = []


lines = []
lines1 = []
lines2 = []

def animate(i):
    #global y0data #????
    tdata.append(t[i])
    y0data.append(y0_x[i])
    y1data.append(y1_x[i])
    
    global ax, lines, lines1, lines2
    for line in lines:
        line.remove()
    for line in lines1:
        line.remove()
    for line in lines2:
        line.remove()
    
    lines1 = ax[0].plot(tdata, y1data, lw = 0.75, color="red")
    lines = ax[0].plot(tdata, y0data, lw = 0.75, color="blue") 
    lines2 = ax[1].plot(tdata, [np.abs(y1data[i]-y0data[i]) for i in range(len(y0data))], color="purple", lw=1)
    return ax
  
ani = FuncAnimation(fig, 
                    animate, 
                    frames = len(t_ev), 
                    interval = 10, 
                    repeat = False, 
                    blit=True)

#plt.tight_layout()
#plt.show()

writer = animation.PillowWriter(fps=30)
ani.save("chaos.png", writer=writer)