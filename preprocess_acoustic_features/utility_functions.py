import mdfreader
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import wavio
import h5py
def mf4_reader(path,filename):
    yop = mdfreader.Mdf(path)
    channel_data = yop.get_channel_data(str(filename))
    return channel_data

def mf4_reader_vib(path):
    yop = mdfreader.Mdf(path)
    #channel_data = yop.get_channel_data()
    channel_data  = yop.keys()
    return channel_data

def channel_data_extractor(path):
    for chan in mf4_reader_vib(path):
        print(str(path) + chan)
        return chan
def h5py_reader_channel(path):
    with h5py.File(path, 'r') as f:
        channel_data=f.keys()
        return list(channel_data)
def h5py_reader(path,chan):
    with h5py.File(path,'r') as f:
        data = np.array(f[str(chan)][:]) #dataset_name is same as hdf5 object name
        return data

def write_wave(data,path,rate):
    wavio.write(path + '.wav', data, rate,sampwidth=2)

def lowpass(rate):
    
    fs = rate       # Sample rate, Hz
    cutoff = 4500    # Desired cutoff frequency, Hz
    trans_width = 500  # Width of transition from pass band to stop band, Hz
    numtaps =400      # Size of the FIR filter.
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0],Hz=fs)

    return taps


def lowpass_vib(rate):
    fs = rate       # Sample rate, Hz
    cutoff = 8000    # Desired cutoff frequency, Hz
    trans_width = 1000  # Width of transition from pass band to stop band, Hz
    numtaps =400     # Size of the FIR filter.
    taps = signal.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0],Hz=fs)
    return taps

def lowpass_butter(rate):
    fs=rate
    cutoff = 12000
    order = 75
    sos = signal.butter(order,cutoff,'low', fs=fs,output='sos')
    return sos

def lowpass_downsample(rate, result):
    filt_lowpass=signal.convolve(result,lowpass(rate))
    filt_lowpass_ds=filt_lowpass[::5]
    return filt_lowpass_ds,rate/5   

def lowpass_ds_vib(rate,result):
    filt_lowpass=signal.convolve(result,lowpass_vib(rate))
    #filt_lowpass=signal.sosfilt(lowpass_butter(rate),result) 
    #print("filt_lowpass",len(filt_lowpass))
    filt_lowpass_ds=filt_lowpass[::3]
    return filt_lowpass_ds,int(rate/3)      
 

def plots(snd,rate,path):
    print('lp_dp',snd.shape)
    f,t,y=signal.stft(snd,fs=rate, window=signal.get_window('hann',1024),nperseg=1024,noverlap=512)
    print('time',len(t))
    print("y complex",y)	
    y = y.T
    y=10*np.log10(np.abs(y),where=np.abs(y)>0)
    print('y',y)
    print('y.shape',y.shape)
    print('max',np.max(y))
    print('min',np.min(y))
    plt.pcolormesh(f, t,y,vmin=-55,vmax=15, cmap='jet')
    plt.title('STFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Time [sec]')
    plt.xlim(f.min(),f.max())
    plt.ylim(t.min(),t.max())
    plt.colorbar()
    #plt.figure()
    #plt.savefig(path,format='jpeg',quality=99)
    plt.show()
    plt.close()




