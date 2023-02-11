import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
from utils import wav2mel, wav2stft, wav2mfcc
from utils_mel import mel_spectrogram

def loadWAV(filename, max_points=32000):
    waveform, sr = torchaudio.load(filename) #sr=16000
    #max_audio = max_frames * hop_size + win_length - hop_size
    #if waveform.shape[1] <= max_points:
    #    waveform = F.pad(waveform, (0, max_points-waveform.shape[1],0,0), mode='constant', value=0)
    #shape of audio is [1, xxxxx]    
    return waveform

class EMA_Dataset:
    
    def __init__(self, path='/data/jiachenlian/data_nsf/emadata', mode='train', **args):
        
        ####record paths for wav file and nema file
        ####train/test = 80%/20%
        self.segment_len = args['segment_len']
        self.wav_paths = []
        self.ema_paths = []
        self.lab_paths = []
        self.ema_npy_paths = []
        self.lab_npy_paths = []
        self.eval = args['vis_kinematics'] or args['vis_gestures']
        self.spk_id_setting = args['spk_id']
        self.mode = mode
        self.fixed_length = args['fixed_length']
        self.threshold = 50 #for ema
        
        if self.mode == 'train':
            if self.spk_id_setting == 'mngu0':
                ema_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_ema_mngu0_train.txt'
                wav_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_wav_mngu0_train.txt'
                lab_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_lab_mngu0_train.txt'
            elif self.spk_id_setting == 'all':
                ema_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_ema_train_all.txt'
                wav_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_wav_train_all.txt'
                lab_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_lab_train_all.txt'
        else:
            ema_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_ema_mngu0_test.txt'
            wav_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_wav_mngu0_test.txt'
            lab_metalist_path = '/data/jiachenlian/data_nsf/emadata/metalist_lab_mngu0_test.txt'
            
        with open(ema_metalist_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.ema_npy_paths.append("/data/jiachenlian/data_nsf/"+line[:-1])
                
        with open(wav_metalist_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.wav_paths.append("/data/jiachenlian/data_nsf/"+line[:-1])
                
        with open(lab_metalist_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.lab_npy_paths.append("/data/jiachenlian/data_nsf/"+line[:-1])
    
        print("###############################all data start#############################################")
        print("spk setting is ", self.spk_id_setting)
        print("# of ema npys is ", len(self.ema_npy_paths)) #4409
        print("# of wavs is ", len(self.wav_paths)) #4409 (full: 4579)
        print("# of lab npys is ", len(self.lab_npy_paths)) #4409
        print("###################################all data end##########################################")


    def __len__(self): #4579
        return len(self.ema_npy_paths)

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        wav_data = loadWAV(wav_path) #[1, T_wav]
        ema_npy_path = self.ema_npy_paths[index]
        lab_npy_path = self.lab_npy_paths[index]
        ema_data = torch.FloatTensor(np.load(ema_npy_path)) #[T_ema, 12]
        lab_data = torch.LongTensor(np.load(lab_npy_path)) #[T_lab]
        lab_data_unique = torch.unique_consecutive(lab_data) #[T_unique]
        
        T_wav = wav_data.shape[1]
        T_ema = ema_data.shape[0]
        
        if not self.eval:
            if self.fixed_length:
                if ema_data.shape[0] >= self.segment_len:
                    start_point_ema = int(random.random()*(ema_data.shape[0]-self.segment_len))
                    start_point_wav = start_point_ema * 80
                else:
                    ema_data = F.pad(ema_data, pad=(0, 0, 0, self.segment_len-ema_data.shape[0]), mode='constant', value=0)
                    wav_data = F.pad(wav_data, pad=(0, 80*self.segment_len-wav_data.shape[1], 0, 0), mode='constant', value=0)
                    start_point_ema = 0
                    start_point_wav = 0
                    
                ema_data = ema_data[start_point_ema:start_point_ema+self.segment_len] #[T_ema, 12]
                wav_data = wav_data[:, start_point_wav:start_point_wav+self.segment_len*80] #[1, T_wav]
                
                if ema_data.shape[0] <= self.segment_len or wav_data.shape[1] <= self.segment_len * 80:
                    ema_data = torch.zeros((self.segment_len, 12))
                    wav_data = torch.zeros((1, self.segment_len*80))
                
                mel_data = mel_spectrogram(y=wav_data, n_fft=1025, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False).squeeze(0).transpose(-1,-2) #[T_mel, 80]
                
        return ema_data.transpose(-1, -2), wav_data.squeeze(0), mel_data
    
    
    
    
class IEEE_Dataset:
    
    def __init__(self, path='data/ieee', mode='train', **args):
        
        ####record paths for wav file and nema file
        ####train/test = 80%/20%

        self.segment_len = args['segment_len']
        self.ema_paths = []
        self.ema_npy_paths = []
        self.lab_npy_paths = []
        self.eval = args['vis_kinematics'] or args['vis_gestures']
        self.mode = mode
        self.fixed_length = args['fixed_length']
        
        if self.mode == 'train':
            ema_metalist_path = 'data/ieee/train_metalist_F0.txt'
        else:
            ema_metalist_path = 'data/ieee/test_metalist_F0.txt'
            
        with open(ema_metalist_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.ema_npy_paths.append(line[:-1])
                
    
        print("###############################all data start#############################################")
        print("# of ema npys is ", len(self.ema_npy_paths)) #1390 for train and 348 for test
        print("###################################all data end##########################################")


    def __len__(self): 
        return len(self.ema_npy_paths)

    def __getitem__(self, index):
        ema_npy_path = self.ema_npy_paths[index]
        ema_data = torch.FloatTensor(np.load(ema_npy_path)) #[T_ema, 24]
        
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
    
    
class rtMRI_Dataset:
    
    def __init__(self, path='data/rtRMI', mode='train', **args):
        
        ####record paths for wav file and nema file
        ####train/test = 80%/20%

        self.segment_len = args['segment_len']
        self.ema_paths = []
        self.ema_npy_paths = []
        self.lab_npy_paths = []
        self.eval = args['vis_kinematics'] or args['vis_gestures']
        self.mode = mode
        self.fixed_length = args['fixed_length']
        
        if self.mode == 'train':
            ema_metalist_path = 'data/rtMRI/train_metalist_all.txt'
        else:
            ema_metalist_path = 'data/rtMRI/test_metalist_all.txt'
            
        with open(ema_metalist_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.ema_npy_paths.append(line[:-1])

        print("###############################all data start#############################################")
        print("# of ema npys is ", len(self.ema_npy_paths)) #1390 for train and 348 for test
        print("###################################all data end##########################################")


    def __len__(self): 
        return len(self.ema_npy_paths)

    def __getitem__(self, index):
        ema_npy_path = self.ema_npy_paths[index]
        ema_data = torch.FloatTensor(np.load(ema_npy_path)) #[T, 340]
#         if ema_data.shape[-1] != 340:
#             ema_data = ema_data[:,:340]
        
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
    

def collate(batch):

    ema_batch, wav_batch, mel_batch, stft_batch, mfcc_batch, wav2vec2_batch, lab_batch = zip(*batch)

    #ema_batch [B, T_ema(variable), 12], actually it is list
    #wav_batch, [B, 1, num_points(variable)], actually it is list
    #mel_batch, [B, T_mel(variable), 80], actually it is 

    
    #ema
    ema_len_batch = torch.IntTensor(
        [len(ema) for ema in ema_batch]
    )
    max_ema_len = torch.max(ema_len_batch)

    ema_batch = torch.stack( #[B, max_ema_T, 12]
        ([
            torch.cat(
                (ema, torch.zeros((max_ema_len - len(ema), 12))),
                dim=0
            ) if len(ema) < max_ema_len else ema
            for ema in ema_batch
        ]), dim=0
    )
    
    #melspec

    mel_len_batch = torch.IntTensor(
        [len(mel) for mel in mel_batch]
    )

    max_mel_len = torch.max(mel_len_batch)
    mel_batch = torch.stack( #[B, max_mel_T, 80]
        ([
            torch.cat(
                (mel, torch.zeros((max_mel_len - len(mel), 80))),
                dim=0
            ) if len(mel) < max_mel_len else mel
            for mel in mel_batch
        ]), dim=0
    )
    
    
    #mono phn label

    lab_len_batch = torch.IntTensor(
        [len(lab_seq) for lab_seq in lab_batch]
    )

    max_label_len = torch.max(lab_len_batch)

    lab_batch = torch.stack( #[B, max_lab_len]
        ([
            torch.cat(
                (label, torch.zeros(max_label_len - len(label))), dim =0
                )if len(label) < max_label_len else label
            for label in lab_batch
        ]), dim=0
    ).long()

    return (
        ema_batch,
        mel_batch, 
        stft_batch,
        mfcc_batch,
        wav2vec2_batch,
        ema_len_batch,
        mel_len_batch, 
        stft_len_batch,
        mfcc_len_batch,
        wav2vec2_len_batch,
        lab_batch,
        lab_len_batch
    )


    

