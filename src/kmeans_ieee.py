import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from kmeans_pytorch import kmeans

def kmeans_ema(spk_id_setting='mngu0'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ema_paths = []
    ema_npy_paths = []
    mode = 'train'
    path = 'data/ieee'
    spk_id_setting = spk_id_setting

    
    ema_metalist_path = 'data/ieee/train_metalist_F0.txt'

    with open(ema_metalist_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            ema_npy_paths.append(line[:-1])

    ema_list = []
    for ema_npy_path in ema_npy_paths:
        ema_data = torch.FloatTensor(np.load(ema_npy_path)) #[T_ieee, 24]
        ema_list.append(ema_data)
    ema_data_huge = torch.cat(ema_list, dim=0) #[T_huge, 24]
    ema_data_huge = ema_data_huge.transpose(0, 1) #[24, T_huge]

    ########################################################
    ####################Then we are going to perform kmeans
    ###########win_size = 41
    ###########inp is [24, T_huge]
    ###########it should be [24, T_huge*41]
    ###########So we pad 20 on both sides
    ema_data_huge_pad = F.pad(ema_data_huge, pad=(20,20,0,0), mode='constant', value=0) #[24, T_huge]
    ####################################

    super_ema_data_huge_list = []
    for t in range(ema_data_huge.shape[1]):
        win_ema = ema_data_huge_pad[:,t:t+41] #[24, 41]
        win_ema = win_ema.reshape(-1) #[24*41=984]
        super_ema_data_huge_list.append(win_ema)

    super_ema_data_huge = torch.stack(super_ema_data_huge_list, dim=0).to(device) #[297878, 984]

    print("shape of original data is:", super_ema_data_huge.shape) #[297878, 984]
    super_ema_data_huge = super_ema_data_huge[:40000]
    print("shape of inp for kmeans is:", super_ema_data_huge.shape) #[297878, 984]
    cluster_ids, cluster_centers = kmeans(X=super_ema_data_huge, num_clusters=40, distance='euclidean', device=device)

    print("kmeans finished!!")
    print("shape of centers of kmeans is", cluster_centers.shape)
    print("shape of ids of kmeans is", cluster_ids.shape)
    np.save("data/kmeans_pretrain/kmeans_centers_ieee.npy", cluster_centers.detach().numpy())
    np.save("data/kmeans_pretrain/kmeans_ids_ieee.npy", cluster_ids.detach().numpy())
    

if __name__ == '__main__':
    kmeans_ema()

