import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from kmeans_pytorch import kmeans
from fast_pytorch_kmeans import KMeans as kmeans_fast
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def kmeans_ema(win_size=None, win_size_con=None, num_gestures=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
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
            
    #random.shuffle(ema_npy_paths)    

    ema_list = []
    for ema_npy_path in ema_npy_paths:
        if not os.path.exists(ema_npy_path):
            continue
        ema_data = torch.FloatTensor(np.load(ema_npy_path)) #[T, 170, 2]
        ema_list.append(ema_data)
    ema_data_huge = torch.cat(ema_list, dim=0) #[T_huge, 340]
    ema_data_huge = ema_data_huge.transpose(0, 1) #[340, T_huge] -> [340, 452849]
    
    T_huge = ema_data_huge.shape[1]
    ema_data_huge_delta = ema_data_huge[:,1:] - ema_data_huge[:,:T_huge-1] #[340, T_huge-1]
    ema_data_huge_delta_energy = (ema_data_huge_delta ** 2).sum(dim=0) #[T_huge-1]
    ema_data_huge_delta_energy_grad = np.gradient(ema_data_huge_delta_energy)
    x_test = np.arange(T_huge-1)
    
#     plt.figure(figsize=(20, 6))
#     plt.plot(x_test[:300], ema_data_huge_delta_energy[:300])
#     #plt.plot(x_test[:300], ema_data_huge_delta_energy_grad[:300])
#     plt.savefig("rr.png")
#     exit()
    
    
    ########################################################
    ####################Then we are going to perform kmeans
    ###########win_size = 41
    ###########inp is [24, T_huge]
    ###########it should be [24, T_huge*41]
    ###########So we pad 20 on both sides
    ema_data_huge_pad = F.pad(ema_data_huge, pad=((win_size-1)//2,(win_size)//2,0,0), mode='constant', value=0) #[340, T_huge+win_size*2]
    ema_data_energy_huge_pad = F.pad(ema_data_huge_delta_energy, pad=((win_size-1)//2,(win_size)//2), mode='constant', value=0) #[T_huge+win_size*2]
    ####################################
    
    super_ema_data_huge_list_con = []
    t = 0
    while t < ema_data_huge.shape[1]:
        win_energy = ema_data_energy_huge_pad[t:t+win_size] 
        max_energy_win = torch.max(win_energy)
        max_energy_index_win = torch.argmax(win_energy)
        
        if max_energy_win > 15:
            win_ema_con = ema_data_huge_pad[:,max_energy_index_win-(win_size_con-1)//2:max_energy_index_win+(win_size_con-1)//2+1]
            if win_ema_con.shape[1] != win_size_con:
                t += win_size
                continue
            win_ema_con = win_ema_con.reshape(-1) #[340*win_size_con=13940]
            super_ema_data_huge_list_con.append(win_ema_con)
        t += win_size
        
#     t = 0
#     while t < ema_data_huge.shape[1]:
#         win_energy = ema_data_energy_huge_pad[t:t+win_size] 
#         max_energy_win = torch.max(win_energy)
#         max_energy_index_win = torch.argmax(win_energy)
        
#         if max_energy_win  < 3:
#             win_ema_con = ema_data_huge_pad[:,t:t+win_size]
#             if win_ema_con.shape[1] != win_size_con:
#                 t += 1
#                 continue
#             win_ema_con = win_ema_con.reshape(-1) #[340*win_size_con=13940]
#             super_ema_data_huge_list_con.append(win_ema_con)
#             t += win_size
#         else:
#             t += 1

    super_ema_data_huge_con = torch.stack(super_ema_data_huge_list_con, dim=0).to(device) #[452849, 13940]
    print(super_ema_data_huge_con.shape)
    
    print("shape of original data is:", super_ema_data_huge_con.shape) #[452849, 13940]
    
    kmeans_model = kmeans_fast(n_clusters=num_gestures, max_iter=1000, mode='euclidean', verbose=1)
    x = super_ema_data_huge_con
    cluster_ids = kmeans_model.fit_predict(x)    
    cluster_centers = kmeans_model.centroids
    print("kmeans finished!!")
    print("shape of centers of kmeans is", cluster_centers.shape)
    print("shape of ids of kmeans is", cluster_ids.shape)
    np.save(f"data/kmeans_pretrain/kmeans_centers_rtMRI_modulation_win_{win_size}_num_gestures_{num_gestures}.npy", cluster_centers.detach().cpu().numpy())
    np.save(f"data/kmeans_pretrain/kmeans_ids_rtMRI_modulation_win_{win_size}_num_gestures_{num_gestures}.npy", cluster_ids.detach().cpu().numpy())    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "main python code")
    parser.add_argument('--win_size', type=int, default=9, help='')
    parser.add_argument('--win_size_con', type=int, default=9, help='')
    parser.add_argument('--num_gestures', type=int, default=20, help='')
    args = parser.parse_args()

    kmeans_ema(win_size=args.win_size, win_size_con=args.win_size_con, num_gestures=args.num_gestures)

