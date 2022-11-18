import numpy as np
from scipy.integrate import solve_ivp
import sys
import pickle

sigma, rho, beta = (10, 28, 8.0/3.0)
total_time_in_sec = int(sys.argv[1]) # seconds
points_per_sec = 200

npoints = total_time_in_sec * points_per_sec

dt = 1 / points_per_sec # seconds_per_point

r0 = [0, 1, 27] 
t_ev = np.linspace(0, total_time_in_sec, npoints, endpoint=True) # time that's being evaluated


def lorenz(t, r, sigma=10, rho=28, beta=8.0/3.0):
    x, y, z = r
    fx = sigma * (y - x)
    fy = rho * x - y - x * z
    fz =  x * y - beta * z
    return np.array([fx, fy, fz], float)

def solving():
    print("About to solve")
    sol = solve_ivp(lorenz, t_span = [0, t_ev[-1]], y0 = r0, t_eval = t_ev, args=(sigma, rho, beta))
    print("\n" + sol.message  + "\n")
    return sol

def main():
	
    sol = solving()
    d = {"sol": sol,
         "args": (sigma, rho, beta)}
    solution_file_path = f"{sys.path[0]}/butterfly_solution"
    solution_file = open(solution_file_path, "wb")
    print(f"\nSaving solution to {solution_file_path}\n")
    pickle.dump(d, solution_file)


if __name__ == "__main__":
    main()