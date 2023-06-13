import os
import time
import sys
import pickle 

total_time = 10 # total time of resulting animation in seconds
fps = 10 
dpi = 200 # resolution of frames saved in dots per inch, 250 is nice, 400 seems overkill
ncores = 8 # number of CPUs to use in the multiprocessing
test = 0 # true (1): shows last frame or false (0): makes full mp4


def main():
    start = time.time()

    os.system(f"cd {sys.path[0]}; python butterfly_eqnsolver.py {total_time}")
    os.system(f"cd {sys.path[0]}; python butterfly_makeframes.py {ncores} {dpi} {fps} {test}")

    frames_folder = f"{sys.path[0]}/butterfly_frames"

    file_name = f"{sys.path[0]}/butterfly_test"

    if os.path.isfile(f"{file_name}.mp4"):
        os.remove(f"{file_name}.mp4")

    os.system(f"ffmpeg -r {fps} -f image2 -s 576x432 -pattern_type glob -i '{frames_folder}/*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p {file_name}.mp4")

    difference = (time.time() - start)

    minutes = int(difference // 60)
    leftover_seconds = int(difference - minutes * 60)
    exit(f"\nIt took: {minutes} min {leftover_seconds} sec")

if __name__ == "__main__":
    main()