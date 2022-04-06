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

from sklearn.cluster import MiniBatchKMeans


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )

def kmeans_ema(**args):
    win_size = args['win_size']
    num_gestures = args['num_gestures']
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
                ######### Filtering ###########
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
    super_ema_data_huge_list_con_ori = []
    
    # # ###############. Peak Part.#################
    for i in range(len(peak_indices_sparse)):
        index = peak_indices_sparse[i]
        if ema_data_huge[:, i - h_win_size : i + h_win_size + 1].shape[1] != win_size:
            continue
        embed = ema_data_huge[:, i - h_win_size : i + h_win_size + 1] #[340, 15]
        if args['hamming_window']:
            filter_window = torch.hamming_window(win_size).reshape(1, -1) #[1, 15]
            embed_after_filter = embed * filter_window #[340, 15]
        elif args['hann_window']:
            filter_window = torch.hann_window(win_size).reshape(1, -1) #[1, 15]
            embed_after_filter = embed * filter_window #[340, 15]
        else:
            embed_after_filter = embed
        super_ema_data_huge_list_con.append(embed_after_filter)
        super_ema_data_huge_list_con_ori.append(embed)
        
        
    ###############. Valley Part. #################
    # for i in range(len(peak_indices_sparse)):
    #     index = peak_indices_sparse[i]
    #     if i + 1 <= len(peak_indices_sparse) - 1:
    #         next_index = peak_indices_sparse[i + 1]
    #     start_indices = torch.linspace(index, next_index-win_size, 3)
    #     for start_index in start_indices:
    #         start_index = int(start_index)
    #         if ema_data_huge[:, start_index: start_index + win_size].shape[1] != win_size:
    #             continue
    #         super_ema_data_huge_list_con.append(ema_data_huge[:, start_index: start_index + win_size])
            
        
    super_ema_data_huge_con = torch.stack(super_ema_data_huge_list_con, dim=0) #[N, 340, 15]
    super_ema_data_huge_con_ori = torch.stack(super_ema_data_huge_list_con_ori, dim=0) #[N, 340, 15]
    print(super_ema_data_huge_con.shape)
    super_ema_data_huge_con = super_ema_data_huge_con.reshape(super_ema_data_huge_con.shape[0], -1).to(device) #[N, 340*15]
    super_ema_data_huge_con_ori = super_ema_data_huge_con_ori.reshape(super_ema_data_huge_con_ori.shape[0], -1).to(device) #[N, 340*15]

    print("shape of original data is:", super_ema_data_huge_con.shape) #[452849, 13940]
    
    #################################################################### kmeans 
    
    feat = super_ema_data_huge_con.detach().cpu().numpy() #(N, d_super)
    feat_ori = super_ema_data_huge_con_ori.detach().cpu().numpy() #(N, d_super)
    
    
    km_model = get_km_model(
        n_clusters=num_gestures,
        init='k-means++',
        max_iter=100,
        batch_size=10000,
        tol=0.0,
        max_no_improvement=100,
        n_init=20,
        reassignment_ratio=0.0,
    )
    
    km_model.fit(feat)
    
    if args['recompute_center']:
        C_np = km_model.cluster_centers_.transpose()
        Cnorm_np = (C_np ** 2).sum(0, keepdims=True)
        dist = (
            (feat ** 2).sum(1, keepdims=True)
            - 2 * np.matmul(feat, C_np)
            + Cnorm_np
        )
        
        km_labels = np.argmin(dist, axis=1)  #(N, )
        feat = super_ema_data_huge_con_ori.detach().cpu()
        km_labels = torch.from_numpy(km_labels)
        
        #Compute Center
        M = torch.zeros(km_labels.max()+1, len(feat))
        M[km_labels, torch.arange(len(feat))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        cluster_centers = torch.mm(M, feat).numpy()
        
    elif args['recompute_centroid']:
        
        C_np = km_model.cluster_centers_ #(num_gestures, D)
        
        feat = feat.transpose()
        featnorm_np = (feat ** 2).sum(0, keepdims=True)
        
        dist = (
            (C_np ** 2).sum(1, keepdims=True)
            - 2 * np.matmul(C_np, feat)
            + featnorm_np
        )  #(num_gestures, N)
        
        print("dist.shape", dist.shape)
        centroid_labels = np.argmin(dist, axis=1)  #(num_gestures, )
        cluster_centers = feat_ori[centroid_labels, :]
        
    else:
        cluster_centers = km_model.cluster_centers_
    
    
    print("kmeans finished!!")
    print("shape of centers of kmeans is", cluster_centers.shape)
    #np.save(f"data/kmeans_pretrain/kmeans_centers_rtMRI_modulation_win_{win_size}_num_gestures_{num_gestures}.npy", cluster_centers.detach().cpu().numpy())
    np.save(f"data/kmeans_pretrain/kmeans_centers_plus_rtMRI_modulation_win_{win_size}_num_gestures_{num_gestures}.npy", cluster_centers)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "main python code")
    parser.add_argument('--win_size', type=int, default=100, help='')
    parser.add_argument('--num_gestures', type=int, default=20, help='')
    parser.add_argument('--hamming_window', action='store_true', help='')
    parser.add_argument('--hann_window', action='store_true', help='')
    parser.add_argument('--recompute_center', action='store_true', help='')
    parser.add_argument('--recompute_centroid', action='store_true', help='')
    
    args = parser.parse_args()

    kmeans_ema(**vars(args))

