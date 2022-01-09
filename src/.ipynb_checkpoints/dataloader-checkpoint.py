import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
from utils import wav2mel, wav2stft, wav2mfcc


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
        self.lab_paths = []
        self.ema_npy_paths = []
        self.lab_npy_paths = []
        self.eval = args['vis_kinematics'] or args['vis_gestures']
        self.spk_id_setting = args['spk_id']
        self.mode = mode
        self.fixed_length = args['fixed_length']

        print("Extract wav, nema, npy info................................../......................")
        
        for spk_id in os.listdir(path):
            if not spk_id.startswith('cin'):
                continue
            if not self.spk_id_setting == 'all':
                if not self.spk_id_setting in spk_id:
                    continue
            spk_id_path = os.path.join(path, spk_id)
            ema_dir = os.path.join(spk_id_path, "nema")
            wav_dir = os.path.join(spk_id_path, "wav")
            lab_dir = os.path.join(spk_id_path, "lab")
           
            utter_id_set = set()
            for ema in os.listdir(ema_dir):
                if ema.endswith('.ema'):
                    ema_path = os.path.join(ema_dir, ema)
                    self.ema_paths.append(ema_path)
                    utter_id = ema_path.split("/")[-1].split('.')[0]
                    utter_id_set.add(utter_id)
                if ema.endswith('.npy'):
                    ema_npy_path = os.path.join(ema_dir, ema)
                    self.ema_npy_paths.append(ema_npy_path)

            for wav in os.listdir(wav_dir):
                if not wav.endswith('.wav'):
                    continue
                wav_path = os.path.join(wav_dir, wav)
                wav_id = wav_path.split("/")[-1].split(".")[0]
                if wav_id not in utter_id_set:  #To make sure that redundant wavs are not included
                    continue
                self.wav_paths.append(wav_path)

            for lab in os.listdir(lab_dir):
                if lab.endswith('.lab'):
                    lab_path = os.path.join(lab_dir, lab)
                    lab_id = lab_path.split("/")[-1].split(".")[0]
                    if lab_id not in utter_id_set:  #To make sure that redundant wavs are not included
                        continue
                    self.lab_paths.append(lab_path)

                if lab.endswith('.npy'):
                    lab_npy_path = os.path.join(lab_dir, lab)
                    self.lab_npy_paths.append(lab_npy_path)

        print("##################################################################################")
        print("spk setting is ", self.spk_id_setting)
        print("# of ema npys is ", len(self.ema_npy_paths)) #4409
        print("# of emas is ", len(self.ema_paths)) #4409
        print("# of wavs is ", len(self.wav_paths)) #4409 (full: 4579)
        print("# of labs is ", len(self.lab_paths)) #4409
        print("# of lab npys is ", len(self.lab_npy_paths)) #4409
        
        #random.shuffle(self.ema_npy_paths)
        train_size = int(0.8 * len(self.ema_npy_paths))
        
        if self.mode == 'train':
            self.ema_npy_paths = self.ema_npy_paths[:train_size]
            self.lab_npy_paths = self.lab_npy_paths[:train_size]
        else:
            self.ema_npy_paths = self.ema_npy_paths[train_size:]
            self.lab_npy_paths = self.lab_npy_paths[train_size:]
        print(self.mode + "_ size is: ", len(self.ema_npy_paths))

        with open("emadata/"+self.mode+"_"+self.spk_id_setting+"_metalist.txt", 'w') as f:
            for ema_npy_path in self.ema_npy_paths:
                f.write(ema_npy_path+'\n')
        with open("emadata/"+self.mode+"_"+self.spk_id_setting+"_metalist_lab.txt", 'w') as f:
            for lab_npy_path in self.lab_npy_paths:
                f.write(lab_npy_path+'\n')
    
    
        print("The Real Test Set to be loaded")
        if not self.mode == 'train':
            self.ema_npy_paths = []
            self.lab_npy_paths = []
            with open("emadata/test_mngu0_metalist.txt") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    self.ema_npy_paths.append(line[:-1])
            with open("emadata/test_mngu0_metalist_lab.txt") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    self.lab_npy_paths.append(line[:-1])
            print("#####Testing size (mngu0) is: ", len(self.ema_npy_paths))
                    
        print("##################################################################################")
        #print("Extract Phoneme Labels(Not Alignment)")

    def __len__(self): #4579
        return len(self.ema_npy_paths)

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        wav_data = loadWAV(wav_path) #[1, num_points]
        mel_data = wav2mel(wav_data) #[T_mel, 80]
        stft_data = wav2stft(wav_data) #[T_stft, 201]
        mfcc_data = wav2mfcc(wav_data) #[T_mfcc, 201]
        #wav2vec2 = wav2vec2(wav_data)
        wav2vec2 = torch.zeros_like(stft_data)
        ema_npy_path = self.ema_npy_paths[index]
        lab_npy_path = self.lab_npy_paths[index]
        ema_data = torch.FloatTensor(np.load(ema_npy_path)) #[T_ema_real, 12]
        lab_data = torch.LongTensor(np.load(lab_npy_path)) #[T_lab]
        lab_data_unique = torch.unique_consecutive(lab_data) #[T_unique]
        
        ####################################
        ########Adopt fixed 500 ema points
        ########We should fix t because t is related to H
        ####################################

        if not self.eval:
            if self.fixed_length:
                if ema_data.shape[0] >= self.segment_len:
                    start_point = int(random.random()*(ema_data.shape[0]-self.segment_len))
                    ema_data = ema_data[start_point:start_point+self.segment_len]
                else:
                    ema_data = F.pad(ema_data, pad=(0, 0, 0, self.segment_len-ema_data.shape[0]), mode='constant', value=0)

        return ema_data, wav_data, mel_data, stft_data, mfcc_data, wav2vec2, lab_data_unique


def collate(batch):

    ema_batch, wav_batch, mel_batch, stft_batch, mfcc_batch, wav2vec2_batch, lab_batch = zip(*batch)

    #ema_batch [B, T_ema(variable), 12], actually it is list
    #wav_batch, [B, 1, num_points(variable)], actually it is list
    #mel_batch, [B, T_mel(variable), 80], actually it is 

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

    stft_len_batch = torch.IntTensor(
        [len(stft) for stft in stft_batch]
    )

    max_stft_len = torch.max(stft_len_batch)
    stft_batch = torch.stack( #[B, max_mel_T, 80]
        ([
            torch.cat(
                (stft, torch.zeros((max_stft_len - len(stft), 201))),
                dim=0
            ) if len(stft) < max_stft_len else stft
            for stft in stft_batch
        ]), dim=0
    )
    
    
    mfcc_len_batch = torch.IntTensor(
        [len(mfcc) for mfcc in mfcc_batch]
    )

    max_mfcc_len = torch.max(mfcc_len_batch)
    mfcc_batch = torch.stack( #[B, max_mel_T, 80]
        ([
            torch.cat(
                (mfcc, torch.zeros((max_mfcc_len - len(mfcc), 39))),
                dim=0
            ) if len(mfcc) < max_mfcc_len else mfcc
            for mfcc in mfcc_batch
        ]), dim=0
    )
    
    
    wav2vec2_len_batch = torch.IntTensor(
        [len(wav2vec2) for wav2vec2 in wav2vec2_batch]
    )

    max_wav2vec2_len = torch.max(wav2vec2_len_batch)
    wav2vec2_batch = torch.stack( #[B, max_mel_T, 80]
        ([
            torch.cat(
                (wav2vec2, torch.zeros((max_wav2vec2_len - len(wav2vec2), 201))),
                dim=0
            ) if len(wav2vec2) < max_wav2vec2_len else wav2vec2
            for wav2vec2 in wav2vec2_batch
        ]), dim=0
    )

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


    

