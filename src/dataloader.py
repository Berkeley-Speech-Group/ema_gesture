import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import sys

def loadWAV(filename, max_points=32000):
    waveform, sr = torchaudio.load(filename) #sr=16000

    #max_audio = max_frames * hop_size + win_length - hop_size
    #if waveform.shape[1] <= max_points:
    #    waveform = F.pad(waveform, (0, max_points-waveform.shape[1],0,0), mode='constant', value=0)
    #shape of audio is [1, xxxxx]    
    
    return waveform

class EMA_Dataset:
    
    def __init__(self, path='emadata', **args):
        
        #record paths for wav file and nema file
        self.segment_len = args['segment_len']
        self.wav_paths = []
        self.ema_paths = []
        self.ema_npy_paths = []
        
        for spk_id in os.listdir(path):
            if not spk_id.startswith('cin'):
                continue
            spk_id_path = os.path.join(path, spk_id)
            ema_dir = os.path.join(spk_id_path, "nema")
            wav_dir = os.path.join(spk_id_path, "wav")
           
            for ema in os.listdir(ema_dir):
                if ema.endswith('.ema'):
                    ema_path = os.path.join(ema_dir, ema)
                    self.ema_paths.append(ema_path)
                if ema.endswith('.npy'):
                    ema_npy_path = os.path.join(ema_dir, ema)
                    self.ema_npy_paths.append(ema_npy_path)
                
            for wav in os.listdir(wav_dir):
                if not wav.endswith('.wav'):
                    continue
                wav_path = os.path.join(wav_dir, wav)
                self.wav_paths.append(wav_path)
        #print(len(self.ema_npy_paths)) #4409
        #print(len(self.ema_paths)) #4409
        #print(len(self.wav_paths)) #4579
               
    def __len__(self): #4579
        return len(self.ema_npy_paths)
    def __getitem__(self, index):
        #wav_path = self.wav_paths[index]
        #waveform = loadWAV(wav_path) #shape is [1, num_points]
        ema_npy_path = self.ema_npy_paths[index]
        ema_data = torch.FloatTensor(np.load(ema_npy_path))
        #print(ema_data.shape) #(T_ema, 12)
        #print(waveform.shape) #(1, T_wav)
        #print(ema_data.shape[0]/waveform.shape[1])

        ####################################
        ########Adopt fixed 500 ema points
        ########We should fix t because t is related to H
        ####################################

        if ema_data.shape[0] >= self.segment_len:
            start_point = int(random.random()*(ema_data.shape[0]-self.segment_len))
            ema_data = ema_data[start_point:start_point+self.segment_len]
        else:
            ema_data = F.pad(ema_data, pad=(0, 0, 0, self.segment_len-ema_data.shape[0]), mode='constant', value=0)
        
        return ema_data


if __name__ == "__main__":
    dataset = EMA_Dataset()
    dataset[0] #wav is [1, 63490], ema is [724, 12] , sr for ema is 182.45
    dataset[1] #wav is [1, 55494], ema is [802, 12] , sr for ema is 231.23
    

