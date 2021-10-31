import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

def loadWAV(filename, max_points):
    audio, _ = torchaudio.load(filename)
    

class EMA_Dataset:
    
    def __init__(self, path='emadata'):
        
        #record paths for wav file and nema file
        self.wav_paths = []
        self.ema_paths = []
        for spk_id in os.listdir(path):
            if not spk_id.startswith('cin'):
                continue
            spk_id_path = os.path.join(path, spk_id)
            ema_dir = os.path.join(spk_id_path, "nema")
            wav_dir = os.path.join(spk_id_path, "wav")
            
            for ema in os.listdir(ema_dir):
                if not ema.endswith('.ema'):
                    continue
                ema_path = os.path.join(ema_dir, ema)
                self.ema_paths.append(ema_path)
                
            for wav in os.listdir(wav_dir):
                if not wav.endswith('.wav'):
                    continue
                wav_path = os.path.join(wav_dir, wav)
                print(wav_path)
                self.wav_paths.append(wav_path)

            
    def __len__(self): #4579
        return len(self.wav_paths)
    def __getitem__(self, index):
        return self.wav_paths[index]
        
        
if __name__ == "__main__":
    dataset = EMA_Dataset()
    print(len(dataset))
    

