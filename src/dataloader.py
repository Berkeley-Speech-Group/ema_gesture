import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

def loadWAV(filename, max_points=25744):
    waveform, _ = torchaudio.load(filename)
    #max_audio = max_frames * hop_size + win_length - hop_size
    if waveform.shape[1] <= max_points:
        waveform = F.pad(waveform, (0, max_points-waveform.shape[1],0,0), mode='constant', value=0)
    #shape of audio is [1, xxxxx]    
    
    return waveform

class EMA_Dataset:
    
    def __init__(self, path='emadata'):
        
        #record paths for wav file and nema file
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
               
            
    def __len__(self): #4579
        return len(self.wav_paths)
    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        waveform = loadWAV(wav_path) #shape is [1, num_points]
        
        ema_npy_path = self.ema_npy_paths[index]
        ema_data = torch.FloatTensor(np.load(ema_npy_path))
        #print(ema_data.shape)
        
        return waveform, ema_data
        
        
if __name__ == "__main__":
    dataset = EMA_Dataset()
    dataset[0]

    

