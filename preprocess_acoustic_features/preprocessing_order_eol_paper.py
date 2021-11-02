import os
import numpy as np
from pathlib import Path
#from sound import sound
import librosa, librosa.display
import matplotlib.pyplot as plt
import math
import json
from utility_functions import mf4_reader,lowpass_downsample,plots, write_wave, channel_data_extractor, mf4_reader_vib,lowpass_ds_vib, h5py_reader_channel,h5py_reader
import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
rootdir = "E:/shaefler_thesis/Eol_mf4_paper"
JSON_PATH = "E:/shaefler_thesis/code/order_EoL_paper.json"
sample_rate=50000
data_duration=25


def save_stft(rootdir, json_path, n_fft=1024, hop_length=512, num_segments=10):
    data = {
        "mapping": [],
        "order": []
    }
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
       
        for file in files:
            path = Path(subdir + '/' + file)
            file = Path(file)
            
            rpm_channel = mf4_reader_vib(path)
            rpm_channel = list(rpm_channel)[0]
          
            rpm=mf4_reader(path,rpm_channel)
            
            rpm = np.floor(rpm)
            start_slice = np.where((rpm>598) & (rpm<601))[0][0]
            stop_slice = np.where((rpm>598) & (rpm<601))[0][-1]

            rpm=rpm[start_slice:stop_slice]
            
            
            for chan in mf4_reader_vib(path):
               
                
                if(chan == "InputSpeed ?"):
                    continue
                
                sample_name = str(file)
               
                print('sample_name',sample_name)
                vib = mf4_reader(path,chan)
                
                                         
                vib=vib[start_slice:stop_slice] 
                
              
                
                
                mat = eng.rpmordermap(matlab.double(vib.tolist()),sample_rate,matlab.double(rpm.tolist()),0.5,'scale','db','Window','hann','amplitude','power');
                mat = np.array(mat).T
                print("mat.shape",mat.shape)
                
                mat_append = np.zeros((770,722),dtype=np.float32)
                print(mat_append[:,:len(mat[1,:])].shape)
                mat_append[:,:len(mat[1,:])] = mat 
                print("mat_append.shape",mat_append.shape)
                
                
                plt.pcolormesh(mat.T, cmap='jet')
                locs, labels = plt.xticks()
                print(locs,labels)
                
                plt.xticks(np.arange(0, 720, step=144),['0','4','8','12', '16'])
                plt.yticks(np.arange(0, 770, step=154),['0', '50','100','150', '200'])
                plt.xlabel('Time')
                plt.ylabel('Order')
                plt.colorbar(format='%+2.f dB')
                plt.title('Order Analysis')
                plt.show()
                
                data['order'].append(mat_append.tolist())
                data['mapping'].append(sample_name)
           

  
    
    print("mappings",data["mapping"])                    
    with open(json_path, "w") as fp:
         json.dump(data, fp, indent=4)

                    
   
                
save_stft(rootdir,JSON_PATH)
            
