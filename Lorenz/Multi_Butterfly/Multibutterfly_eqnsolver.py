import numpy as np
from scipy.integrate import solve_ivp
import sys
import pickle

import time

sigma, rho, beta = (10, 28, 8.0/3.0)
total_time_in_sec = 100 #int(sys.argv[1]) # seconds
points_per_sec = 400

npoints = total_time_in_sec * points_per_sec

dt = 1 / points_per_sec # seconds_per_point

dIC = np.array([0, 0.00001, 0])
r0 = np.array([0, 1, 27])
r1 = r0 + dIC
r2 = r1 + dIC
r3 = r2 + dIC
num_IC = 4


t_ev = np.linspace(0, total_time_in_sec, npoints, endpoint=True) # time that's being evaluated


def lorenz(t, r, sigma=10, rho=28, beta=8.0/3.0):
    x, y, z = r
    fx = sigma * (y - x)
    fy = rho * x - y - x * z
    fz =  x * y - beta * z
    return np.array([fx, fy, fz], float)

def solving():
    

    print("About to solve")
    sol_0 = solve_ivp(lorenz, t_span = [0, t_ev[-1]], y0 = r0, t_eval = t_ev, args=(sigma, rho, beta))
    sol_1 = solve_ivp(lorenz, t_span = [0, t_ev[-1]], y0 = r1, t_eval = t_ev, args=(sigma, rho, beta))
    sol_list = np.array([sol_0.y, sol_1.y])
    print("\n" + sol_0.message  + "\n")
    print(f"{sol_list=}")
    return sol_list

def main():
	
    solution = solving()
    d = {"sol": solution,
         "time": t_ev,
         "args": (sigma, rho, beta)}
    solution_file_path = f"{sys.path[0]}/Multibutterfly_solution"
    solution_file = open(solution_file_path, "wb")
    print(f"\nSaving solution to {solution_file_path}\n")
    pickle.dump(d, solution_file)


if __name__ == "__main__":
    start = time.time()
    main()
    print(time.time()-start)