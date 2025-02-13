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
# from pydub import Segment
from pydub.utils import make_chunks
from scipy.io import wavfile
from matplotlib import pyplot as plt
from PIL import Image
import soundfile as sf
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--path',type=str)
args = parser.parse_args()

# postfix = "_22k"

def graph_spectrogram(wav_file):
        print(len(glob.glob('fix_len_audio_trim/segments/'+wav_file+'/*.wav')))
        if len(glob.glob('fix_len_audio_trim/segments/'+wav_file+'/*.wav')) < 7:
            print(wav_file, ' less than 7')
            return
        if os.path.exists('fix_len_audio_trim/spec/'+wav_file):
            return
        os.makedirs('fix_len_audio_trim/spec/'+wav_file,exist_ok=True)
        # file_path = "../Stroke_data/7seg/audio_segment'+postfix+'/"+wav_file+".wav"
        # filename = 'Haunting_song_of_humpback_whales.wav'
        
        # trim silent edges
        # whale_songs, _ = librosa.effects.trim(y)
        for idx in range(7):
            
            y, sr = librosa.load('fix_len_audio_trim/segments/'+wav_file+'/%02d'%(idx+1)+'.wav')
            whale_song, _ = librosa.effects.trim(y)

            n_fft = 2048 

            hop_length = 512
            D = np.abs(librosa.stft(whale_song, n_fft=n_fft,  hop_length=hop_length))

            DB = librosa.amplitude_to_db(D, ref=np.max)
            n_mels = 128
            S = librosa.feature.melspectrogram(y=whale_song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_DB = librosa.power_to_db(S, ref=np.max)

            librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length)
            # librosa.display.specshow(S_DB, y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length)


            plt.axis('off')
            plt.savefig('fix_len_audio_trim/spec/'+wav_file+'/%02d'%(idx+1)+'.png',dpi=300,  # Dots per inch
                            bbox_inches='tight',
                            pad_inches=0)
        
        
# f = os.listdir("../Stroke_data/7seg/audio_segment")
# for item in f:
graph_spectrogram(args.path)
