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

def kmeans_ema(win_size=None,  num_gestures=None):
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
    
    ########################################################
    ####################Then we are going to perform kmeans
    ###########win_size = 41
    ###########inp is [24, T_huge]
    ###########it should be [24, T_huge*41]
    ###########So we pad 20 on both sides
    #ema_data_huge_pad = F.pad(ema_data_huge, pad=((win_size-1)//2,(win_size)//2,0,0), mode='constant', value=0) #[340, T_huge+win_size*2]
    #ema_data_energy_huge_pad = F.pad(ema_data_huge_delta_energy, pad=((win_size-1)//2,(win_size)//2), mode='constant', value=0) #[T_huge+win_size*2]
    ####################################
    
    
    ######################################################## zero array ##########################################################
    test_arr = ema_data_huge_delta_energy.clone()
    for i in range(len(test_arr)):
        if test_arr[i] < 10:
            test_arr[i] = 0
#     x = np.arange(len(test_arr))
#     plt.figure(figsize=(20.1, 6))
#     plt.plot(x[:500], test_arr[:500])
#     plt.savefig("test.png")
#     exit()
    
            
    ######################################################## Find Peak ##########################################################
    test_arr2 = test_arr.clone()
    peak_indices = []
    peak_indices_set = set()
    h_win_size = (win_size - 1) // 2


    ###############################################    Locate Peak Indices    ###########################################
    for i in range(len(test_arr2)):
        if i > 1 and i < len(test_arr2) - 1:
            if test_arr2[i] > test_arr2[i + 1] and test_arr2[i] > test_arr2[i - 1]:
                peak_indices.append(i)
                peak_indices_set.add(i)

    peak_indices_sparse = []
    peak_indices_sparse_set = set()

    for index in peak_indices:
        flag = False
        if index + h_win_size <= len(test_arr2) -1 and index - h_win_size >= 0:
            for j in range(index - h_win_size, index + h_win_size + 1):
                if j == index:
                    continue
                if j in peak_indices_set:
                    flag = True
                    break
            if not flag:
                peak_indices_sparse.append(index)
                peak_indices_sparse_set.add(index)
                
#     y = test_arr2.clone()
#     x = np.arange(len(y))
#     for i in range(len(y)):
#         if i not in peak_indices_sparse_set:
#             y[i] = 0

#     plt.figure(figsize=(20.1, 6))
#     plt.plot(x[:5000], y[:5000])
#     plt.xticks()
#     plt.yticks()
#     #plt.xlim(0,200)
#     plt.savefig("test1.png")
#     exit()
                
    
    ###############################################    collect ema kinematics    ###########################################
    super_ema_data_huge_list_con = []
    
    ###############. Peak Part. #################
    for i in range(len(peak_indices_sparse)):
        index = peak_indices_sparse[i]
        if ema_data_huge[:, i - h_win_size : i + h_win_size + 1].shape[1] != win_size:
            continue
        super_ema_data_huge_list_con.append(ema_data_huge[:, i - h_win_size : i + h_win_size + 1])
        
        
    ###############. Valley Part. #################
    for i in range(len(peak_indices_sparse)):
        index = peak_indices_sparse[i]
        if i + 1 <= len(peak_indices_sparse) - 1:
            next_index = peak_indices_sparse[i + 1]
        start_indices = torch.linspace(index, next_index-win_size, 10)
        for start_index in start_indices:
            start_index = int(start_index)
            if ema_data_huge[:, start_index: start_index + win_size].shape[1] != win_size:
                continue
            super_ema_data_huge_list_con.append(ema_data_huge[:, start_index: start_index + win_size])
            
        
    super_ema_data_huge_con = torch.stack(super_ema_data_huge_list_con, dim=0) #[N, 340, 15]
    print(super_ema_data_huge_con.shape)
    super_ema_data_huge_con = super_ema_data_huge_con.reshape(super_ema_data_huge_con.shape[0], -1).to(device) #[N, 340*15]

    print("shape of original data is:", super_ema_data_huge_con.shape) #[452849, 13940]
    
    
    #################################################################### kmeans ##############################################################################
    
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
    parser.add_argument('--win_size', type=int, default=15, help='')
    parser.add_argument('--num_gestures', type=int, default=20, help='')
    args = parser.parse_args()

    kmeans_ema(win_size=args.win_size, num_gestures=args.num_gestures)

