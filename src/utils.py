import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

def vis_kinematics(model, **args):
    ema_data = np.load(args['test_ema_path']) #[t, 12]
    ema_id = args['test_ema_path'].split("/")[-1][:-4]
    x = np.arange(ema_data.shape[0])

    fig, axs = plt.subplots(12)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    fig.suptitle(ema_id+"_ori")
    for i in range(12):
        axs[i].plot(x, ema_data[:,i],c=colors[i//2])
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])

    # axs[i].plot(x, ema_data[:,i])
    # plt.plot(x, ema_data[:,1])
    # plt.plot(x, ema_data[:,2])
    # plt.plot(x, ema_data[:,3])
    # plt.plot(x, ema_data[:,4])
    # plt.plot(x, ema_data[:,5])
    # plt.plot(x, ema_data[:,6])
    # plt.plot(x, ema_data[:,7])
    # plt.plot(x, ema_data[:,8])
    # plt.plot(x, ema_data[:,9])
    # plt.plot(x, ema_data[:,10])
    # plt.plot(x, ema_data[:,11])

    plt.savefig(os.path.join(args['save_path'], "ori_"+ema_id+".png"))
    plt.clf()
    return None

def vis_gestures(model, **args):
    return None