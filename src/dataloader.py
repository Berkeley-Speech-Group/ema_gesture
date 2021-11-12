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
    
    def __init__(self, path='emadata', mode='train', **args):
        
        ####record paths for wav file and nema file
        ####train/test = 80%/20%

        self.segment_len = args['segment_len']
        self.wav_paths = []
        self.ema_paths = []
        self.ema_npy_paths = []
        self.eval = args['vis_kinematics'] or args['vis_gestures']
        self.spk_id_setting = args['spk_id']
        self.mode = mode
        
        for spk_id in os.listdir(path):
            if not spk_id.startswith('cin'):
                continue
            if not self.spk_id_setting == 'all':
                if not self.spk_id_setting in spk_id:
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

        random.shuffle(self.ema_npy_paths)
        train_size = int(0.8 * len(self.ema_npy_paths))
        
        if self.mode == 'train':
            self.ema_npy_paths = self.ema_npy_paths[:train_size]
        else:
            self.ema_npy_paths = self.ema_npy_paths[train_size:]
        print(self.mode + "_ size is: ", len(self.ema_npy_paths))

        with open("emadata/"+self.mode+"_metalist.txt", 'w') as f:
            for ema_npy_path in self.ema_npy_paths:
                f.write(ema_npy_path+'\n')

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

        if not self.eval:
            if ema_data.shape[0] >= self.segment_len:
                start_point = int(random.random()*(ema_data.shape[0]-self.segment_len))
                ema_data = ema_data[start_point:start_point+self.segment_len]
            else:
                ema_data = F.pad(ema_data, pad=(0, 0, 0, self.segment_len-ema_data.shape[0]), mode='constant', value=0)
        return ema_data

    

