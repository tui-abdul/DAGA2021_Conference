import os
import numpy as np
from pathlib import Path
import librosa, librosa.display
import matplotlib.pyplot as plt
import math
import json
from utility_functions import mf4_reader,lowpass_downsample,plots, write_wave, channel_data_extractor, mf4_reader_vib,lowpass_ds_vib
rootdir = "E:/shaefler_thesis/Eol_mf4_paper"
JSON_PATH = "E:/shaefler_thesis/code/stft_EoL_no_seg_paper.json"



def save_stft(rootdir, json_path, n_fft=1024, hop_length=512):
    data = {
        "mapping": [],
        "stft": []
    }
    
    for subdir, dirs, files in os.walk(rootdir):
        
        for file in files:
            path = Path(subdir + '/' + file)
            file = Path(file)
          
            sample_name = str(file)
            for chan in mf4_reader_vib(path):
                if (chan == 'KS-Getr'):

                    name = str(path) + chan
                    

                    
                    rpm_channel = mf4_reader_vib(path)
                    rpm_channel = list(rpm_channel)[0]
                    rpm=mf4_reader(path,rpm_channel)
                 
                    rpm = np.floor(rpm)
                    start_slice = np.where((rpm>598) & (rpm<601))[0][0]
                    stop_slice = np.where((rpm>598) & (rpm<601))[0][-1]
                    rpm=rpm[start_slice:stop_slice]
                  
                    
                
                    vib = mf4_reader(path,chan)
                    vib=vib[start_slice:stop_slice]
                    
                  
                    
                    stft = librosa.stft(vib,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window='hann')
                    stft = stft.T
                    stft = librosa.amplitude_to_db(np.abs(stft))
                    
                    stft = stft[:1915,:]
                    
                    
                    data['stft'].append(stft.tolist())
                    data['mapping'].append(sample_name)
                  
    print("mappings",data["mapping"])                    
    with open(json_path, "w") as fp:
         json.dump(data, fp, indent=4)

                    

save_stft(rootdir,JSON_PATH)
            
