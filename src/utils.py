import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def vis_H(model, **args):
    ema_data = np.load(args['test_ema_path']) #[t, 12]
    ema_ori, ema_hat, latent_H, _, _ = model(torch.FloatTensor(ema_data).unsqueeze(0).to(device))
    #print(latent_H.shape) #[1,1,num_gestures, t]
    latent_H = latent_H.squeeze().squeeze().detach().numpy()
    #print(latent_H)
    #ax = sns.heatmap(latent_H, linewidth=0.5)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    latent_H = latent_H[:,:100]
    im = ax.imshow(latent_H, cmap='hot', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(args['save_path'], 'latent_H'+"_"+".png"))
    plt.clf()

def vis_kinematics(model, **args):
    ######################################
    ############The Original Data
    #####################################
    ema_data = np.load(args['test_ema_path']) #[t, 12]
    ema_id = args['test_ema_path'].split("/")[-1][:-4]
    #draw_kinematics(ema_data, mode=ema_id+'_ori', **args)

    ######################################
    ############Reconstruction
    #####################################
    ema_ori, ema_hat,_,_,_ = model(torch.FloatTensor(ema_data).unsqueeze(0).to(device))
    ema_data_hat = ema_hat.squeeze(0).transpose(0,1).detach().numpy()

    draw_kinematics(ema_data, ema_data_hat, mode='kinematics', title=ema_id+'ori_rec', **args) 
    

def draw_kinematics(ema_data, ema_data_hat, mode, title, **args):
    
    x = np.arange(ema_data.shape[0])
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(title)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    outer = gridspec.GridSpec(6, 1, wspace=0.2, hspace=0.2)
    labels = ['tongue dorsum', 'tongue blade', 'tongue tip', 'lower incisor', 'upper lip', 'lower lip']
    for i in range(6):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            ax.plot(x, ema_data[:,i*2+j],c=colors[i], linestyle='dashed', label='ori')
            if mode == 'kinematics':
                ax.plot(x, ema_data_hat[:,i*2+j],c=colors[i], label='rec')
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
    plt.savefig(os.path.join(args['save_path'], title+"_"+".png"))
    plt.clf()

def vis_gestures(model, **args):
    gestures = model.conv_decoder.weight #[num_pellets, 1, num_gestures, win_size]
    for i in range(args['num_gestures']):
        gesture_index = i
        draw_kinematics(gestures[:,0,gesture_index,:].transpose(0,1).detach().numpy(), None, mode='gesture', title='gesture_'+str(gesture_index), **args)
    