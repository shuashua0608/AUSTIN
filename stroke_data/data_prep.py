import subprocess
import os
import sys
import cv2
import glob
import shutil
import json
import csv
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
from scipy.io import wavfile
from matplotlib import pyplot as plt
from PIL import Image

def graph_spectrogram(wav_file):
    file_path = "RawData/Audio/"+wav_file+".wav"
    # filename = 'Haunting_song_of_humpback_whales.wav'
    y, sr = librosa.load(file_path)
    # trim silent edges
    whale_song, _ = librosa.effects.trim(y)
    librosa.display.waveshow(whale_song, sr=sr)

    n_fft = 2048 

    hop_length = 512
    D = np.abs(librosa.stft(whale_song, n_fft=n_fft,  hop_length=hop_length))

    DB = librosa.amplitude_to_db(D, ref=np.max)
    # librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    n_mels = 128
    S = librosa.feature.melspectrogram(whale_song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length)


    plt.axis('off')
    plt.savefig('Feature/Spectrograms/'+wav_file+'.png',dpi=300,  # Dots per inch
                    bbox_inches='tight',
                    pad_inches=0)


# if not os.path.exists(os.curdir+"/Feature/Spectrograms/"):
#     os.makedirs(os.curdir+"/Feature/Spectrograms/")
#     os.makedirs(os.curdir+"/Feature/Frames/")


#Extract wav-> RawData/Video/* ~ RawData/Audio/*
# if not os.path.exists(os.curdir+"/Audio/"):
#     os.makedirs(os.curdir+"/Audio/")
# for root, dirs, files in os.walk(os.curdir+"/Raw_Video/"):
#     for filename in files:
#         # print(filename)
#         command = "ffmpeg -i "+os.curdir+"/Raw_Video/"+filename+" -ab 160k -ac 2 -ar 44100 -vn " + os.curdir+"/Audio/"+filename[:-4]+".wav"
#         subprocess.call(command, shell=True)

# Video Tracking & Cropping
f = os.listdir(os.curdir+"/Raw_Video/")
for item in f:
    print(item[:-4])
    # video_filepath = line.strip("\n")
    os.chdir(os.curdir+"/faceTracking/")
    os.system("python framerun.py -v "+item[:-4])
    os.chdir("../")
          

# f = os.listdir(os.curdir+"/RawData/Audio/")
# for item in f:
#     # os.system("python spec.py -v "+item[:-4])
#     graph_spectrogram(item[:-4])
