import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from kmeans_pytorch import kmeans
from vq import *
from tqdm import tqdm

def kmeans_ema(spk_id_setting='mngu0'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wav_paths = []
    ema_paths = []
    ema_npy_paths = []
    mode = 'train'
    path = 'data/emadata'
    spk_id_setting = spk_id_setting

    for spk_id in os.listdir(path):
        if not spk_id.startswith('cin'):
            continue
        if not spk_id_setting == 'all':
            if not spk_id_setting in spk_id:
                continue
        spk_id_path = os.path.join(path, spk_id)
        ema_dir = os.path.join(spk_id_path, "nema")
        wav_dir = os.path.join(spk_id_path, "wav")
        
        for ema in os.listdir(ema_dir):
            if ema.endswith('.ema'):
                ema_path = os.path.join(ema_dir, ema)
                ema_paths.append(ema_path)
            if ema.endswith('.npy'):
                ema_npy_path = os.path.join(ema_dir, ema)
                ema_npy_paths.append(ema_npy_path)
            
        for wav in os.listdir(wav_dir):
            if not wav.endswith('.wav'):
                continue
            wav_path = os.path.join(wav_dir, wav)
            wav_paths.append(wav_path)

    #random.shuffle(ema_npy_paths)
    train_size = int(0.8 * len(ema_npy_paths))
    
    if mode == 'train':
        ema_npy_paths = ema_npy_paths[:train_size]
    else:
        ema_npy_paths = ema_npy_paths[train_size:]
    print(mode + "_ size is: ", len(ema_npy_paths))

    ema_list = []
    for ema_npy_path in ema_npy_paths:
        ema_data = torch.FloatTensor(np.load(ema_npy_path)) #[T_ema, 12]
        ema_list.append(ema_data)
    ema_data_huge = torch.cat(ema_list, dim=0) #[716316, 12]
    ema_data_huge = ema_data_huge.transpose(0, 1) #[12, 716316]

    ########################################################
    ####################Then we are going to perform kmeans
    ###########win_size = 41
    ###########inp is [12, 716316]
    ###########it should be [12, 716316*41]
    ###########So we pad 20 on both sides
    ema_data_huge_pad = F.pad(ema_data_huge, pad=(20,20,0,0), mode='constant', value=0) #[12, 716356]
    ####################################

    super_ema_data_huge_list = []
    for t in range(ema_data_huge.shape[1]):
        win_ema = ema_data_huge_pad[:,t:t+41] #[12, 41]
        win_ema = win_ema.reshape(-1) #[12*41=492]
        super_ema_data_huge_list.append(win_ema)

    super_ema_data_huge = torch.stack(super_ema_data_huge_list, dim=0).to(device) #[716316, 492]

    print("shape of original data is:", super_ema_data_huge.shape) #[716316, 492]
    super_ema_data_huge = super_ema_data_huge[:50000]
    print("shape of inp for kmeans is:", super_ema_data_huge.shape) #[716316, 492]
    
    return super_ema_data_huge

class VQ_Dataset:
    def __init__(self, data):
        super().__init__()
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
        
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    super_ema_data_huge = kmeans_ema()
    vq_dataset = VQ_Dataset(super_ema_data_huge)
    vq_dataloader = torch.utils.data.DataLoader(dataset=vq_dataset, batch_size=100, shuffle=True)
    vq_vae = VQ_VAE2().to(device)
    for e in tqdm(range(100)):
        for i, batch in enumerate(vq_dataloader):
            loss_vq, _, _, = vq_vae(batch)
            print(vq_vae._embedding.weight)
            
    np.save("data/kmeans_pretrain/kmeans_centers_vq1.npy", vq_vae._embedding.weight.detach().cpu().numpy())

