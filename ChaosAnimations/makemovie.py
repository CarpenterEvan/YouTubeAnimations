import os 
os.system(" ".join(['/opt/anaconda3/bin/ffmpeg', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '1280x960', '-pix_fmt', 'rgba', '-r', '30', '-loglevel', 'error', '-i', 'pipe:', '-vcodec', 'h264', '-pix_fmt', 'yuv420p', '-y', 'chaos_test.mov']))