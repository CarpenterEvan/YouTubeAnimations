# YouTube Animations



https://user-images.githubusercontent.com/915120/201575829-ef1d92e5-d9a3-42c7-bf97-6c36ad0d84b5.mp4


This is rather delicate, and I haven't spent too much time yet making it portable. So if you are someone that wants the code to work on your computer, feel free to try it out!
Just know that you might need to have some patience with me :)


The general layout I try to have going in each of the subfolders is a "eqnsolve" file that solves the relevant equations, a "makeframes" file that plots and saves each of the frames, and a "makemovie" file that runs the first two, then stitches the resulting frames together with ffmpeg. 

three sepearate files -- why? 

Because I want to use multiprocessing file to speed up the process of making frames, but when using it in the same file as the equation solving and frame-stitching, it would re-solve the equations and re-stitch the frames with every new parallel process it creates. 

I'm sure there is a way to properly control which information is shared across processses but I don't know how to do that yet, so three files for now :) 

```sh
docker compose run yt-animations
```
