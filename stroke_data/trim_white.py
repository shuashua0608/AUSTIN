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
import soundfile as sf

def graph_spectrogram(wav_file):
    file_path = "./denoise_Audio_44k/"+wav_file+".wav"
    # filename = 'Haunting_song_of_humpback_whales.wav'
    # y, sr = sf.read(file_path)
    # print(y.shape)
    y, sr = librosa.load(file_path,sr=44100)
    # print(np.max(y))
    S = np.abs(librosa.stft(y))
    # print(np.mean(librosa.power_to_db(S**2, ref=np.max)))
    # trim silent edges
    whale_songs, _ = librosa.effects.trim(y,top_db=18,frame_length=256)
    # print(y.shape, whale_songs.shape)
    print(librosa.get_duration(y, sr=sr), librosa.get_duration(whale_songs, sr=sr))
    sf.write('./denoise_Audio_44k_trim/'+wav_file+'.wav', data=whale_songs, samplerate=44100)
        
        
# f = os.listdir("./Audio_44k")
# for item in f:
graph_spectrogram(sys.argv[1])
# print(sys.argv[1])