import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from kmeans_pytorch import kmeans
from fast_pytorch_kmeans import KMeans as kmeans_fast
from tqdm import tqdm

def kmeans_ema():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ema_paths = []
    ema_npy_paths = []
    mode = 'train'
    path = 'data/rtMRI'
    
    ema_metalist_path = 'data/rtMRI/train_metalist_all.txt'

    with open(ema_metalist_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            ema_npy_paths.append(line[:-1])
            
    random.shuffle(ema_npy_paths)    

    ema_list = []
    for ema_npy_path in ema_npy_paths:
        ema_data = torch.FloatTensor(np.load(ema_npy_path)) #[T, 170, 2]
        ema_data = ema_data.reshape(ema_data.shape[0], -1) #[T, 340]
        ema_list.append(ema_data)
    ema_data_huge = torch.cat(ema_list, dim=0) #[T_huge, 340]
    ema_data_huge = ema_data_huge.transpose(0, 1) #[340, T_huge] = [340, 452849]
    

    ########################################################
    ####################Then we are going to perform kmeans
    ###########win_size = 41
    ###########inp is [24, T_huge]
    ###########it should be [24, T_huge*41]
    ###########So we pad 20 on both sides
    ema_data_huge_pad = F.pad(ema_data_huge, pad=(20,20,0,0), mode='constant', value=0) #[340, T_huge]
    ####################################
    

    super_ema_data_huge_list = []
    for t in tqdm(range(ema_data_huge.shape[1])):
        win_ema = ema_data_huge_pad[:,t:t+41] #[340, 41]
        win_ema = win_ema.reshape(-1) #[340*41=13940]
        super_ema_data_huge_list.append(win_ema)
        if t == 100000:
            break

    super_ema_data_huge = torch.stack(super_ema_data_huge_list, dim=0).to(device) #[452849, 13940]
    

    print("shape of original data is:", super_ema_data_huge.shape) #[452849, 13940]
    
    #     previous methods
    #     super_ema_data_huge = super_ema_data_huge[:40000]
    #     print("shape of inp for kmeans is:", super_ema_data_huge.shape) #[452849, 13940]
    #     cluster_ids, cluster_centers = kmeans(X=super_ema_data_huge, num_clusters=40, distance='euclidean', device=device)

    #     print("kmeans finished!!")
    #     print("shape of centers of kmeans is", cluster_centers.shape)
    #     print("shape of ids of kmeans is", cluster_ids.shape)
    #     np.save("data/kmeans_pretrain/kmeans_centers_rtMRI.npy", cluster_centers.detach().numpy())
    #     np.save("data/kmeans_pretrain/kmeans_ids_rtMRI.npy", cluster_ids.detach().numpy())
    
    kmeans_model = kmeans_fast(n_clusters=40, mode='euclidean', verbose=1)
    x = super_ema_data_huge
    cluster_ids = kmeans_model.fit_predict(x)    
    cluster_centers = kmeans_model.centroids
    print("kmeans finished!!")
    print("shape of centers of kmeans is", cluster_centers.shape)
    print("shape of ids of kmeans is", cluster_ids.shape)
    np.save("data/kmeans_pretrain/kmeans_centers_rtMRI.npy", cluster_centers.detach().cpu().numpy())
    np.save("data/kmeans_pretrain/kmeans_ids_rtMRI.npy", cluster_ids.detach().cpu().numpy())    
    

if __name__ == '__main__':
    kmeans_ema()

