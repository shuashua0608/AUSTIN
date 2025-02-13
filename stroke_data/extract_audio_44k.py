import subprocess
import os
import sys

if not os.path.exists(os.curdir+"/Audio_44k/"):
    os.makedirs(os.curdir+"/Audio_44k/")
for root, dirs, files in os.walk(os.curdir+"/Raw_Video/"):
    for filename in files:
        # print(filename)
        command = "ffmpeg -i "+os.curdir+"/Raw_Video/"+filename+" -ab 160k -ac 2 -ar 44100 -vn " + os.curdir+"/Audio_44k/"+filename[:-4]+".wav"
        subprocess.call(command, shell=True)