import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def vis_kinematics(model, **args):
    ######################################
    ############The Original Data
    #####################################
    ema_data = np.load(args['test_ema_path']) #[t, 12]
    draw_kinematics(ema_data, mode='ori', **args)

    ######################################
    ############Reconstruction
    #####################################
    _, ema_hat = model(torch.FloatTensor(ema_data).unsqueeze(0).to(device))
    ema_data_hat = ema_hat.squeeze(0).transpose(0,1).detach().numpy()
    draw_kinematics(ema_data_hat, mode='rec', **args) 
    

def draw_kinematics(ema_data, mode, **args):
    
    ema_id = args['test_ema_path'].split("/")[-1][:-4]
    x = np.arange(ema_data.shape[0])

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(ema_id+"_"+mode)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    outer = gridspec.GridSpec(6, 1, wspace=0.2, hspace=0.2)
    labels = ['tongue dorsum', 'tongue blade', 'tongue tip', 'lower incisor', 'upper lip', 'lower lip']
    for i in range(6):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)

        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            ax.plot(x, ema_data[:,i*2+j],c=colors[i])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if j == 0:
                ax.set_ylabel(labels[i]+' x',rotation=0)
            else:
                ax.set_ylabel(labels[i]+' y',rotation=0)
            ax.yaxis.set_label_coords(-0.05,0.5)

    plt.savefig(os.path.join(args['save_path'], mode+"_"+ema_id+".png"))
    plt.clf()

def vis_gestures(model, **args):
    return None