# convert the videos to mp3

import os
import subprocess

files = os.listdir("video")

for file in files:
    # remove .mp4
    name_without_ext = file.replace(".mp4", "")
    
    # split on '- Tutorial #'
    title_part, tutorial_part = name_without_ext.split(" - Tutorial #")
    
    tutorial_number = tutorial_part
    title = title_part.split(" _ ")[0]
    
    print(tutorial_number, title)
    subprocess.run(["ffmpeg", "-i",f"video/{file}",f"audios/{tutorial_number}_{title}.mp3"])
