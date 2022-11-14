import os
import time
import sys
start = time.time()

os.system(f"cd {sys.path[0]}; ls; python butterfly_solver.py")
os.system(f"cd {sys.path[0]}; python butterfly_makeframes.py")
os.system(f"cd {sys.path[0]}; python makemovie.py")

difference = (time.time()-start)

minutes = int(difference//60)
leftover_seconds = int(difference - minutes*60)
exit(f"\nIt took: {minutes} min {leftover_seconds} sec")