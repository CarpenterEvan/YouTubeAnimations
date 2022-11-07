import os
import time
os.system("python butterfly_framesave.py")
import butterfly_framesave
start = time.time()
fps = butterfly_framesave.fps
os.system(f"ffmpeg -r {fps} -f image2 -s 576x432 -i butterfly_frames/_img%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p butterfly_test.mp4")
exit(f"\nIt took: {(time.time()-start):.2f} seconds to compile the frames")