import os
import glob
import sys
import pickle 



video_parameters_path = f"{sys.path[0]}/video_parameters"
print(f"\nOpening video parameters from {video_parameters_path}\n")
video_parameters = pickle.load(open(video_parameters_path, "rb"))

frames_folder = video_parameters["frames_folder"]
file_name = video_parameters["file_name"]
fps = video_parameters["fps"]



if os.path.isfile(f"{sys.path[0]}/{file_name}.mp4"):
	os.remove(f"{file_name}.mp4")



os.system(f"ffmpeg -r {fps} -f image2 -s 576x432 -pattern_type glob -i '{frames_folder}/*.png' -vcodec libx264 -crf 25  -pix_fmt yuv420p {file_name}.mp4")

