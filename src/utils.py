import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    #delta = fft_length - hop_length
    #if (len(x) - delta) % hop_length != 0:
    #    pad_width = (len(x) // hop_length + 1) * hop_length + delta - len(x)
    #    x = np.pad(x, (0, int(pad_width)), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)  

def wav2mel(wav):
    #input size of wav should be [1, length]
    mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
    min_level = np.exp(-100 / 20 * np.log(10))
    b, a = butter_highpass(30, 16000, order=5)

    wav = signal.filtfilt(b, a, wav.reshape(-1).numpy())
    D = pySTFT(wav).T
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)  
    mel_spec = torch.FloatTensor(S) #shape should be [T, 80]
    return mel_spec

def draw_mel(mels, mode, title):
    #shape of mels should be [B,D=80, T]
    mels = F.relu(mels)
    mels = mels.transpose(-1, -2)

    for i in range(len(mels)):
        #mel_i = mels[i]
        #np.save("save_models/dsvae2/"+mode+"_mel_"+str(i)+".npy", mel_i.cpu().detach().numpy())
        plt.imshow(mels[i][:,:].transpose(0,1).cpu().detach().numpy())
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.title(title, fontsize=30)
        plt.savefig("save_models/test/"+mode+"_mel_"+str(i)+".png")
        plt.clf()

def ema2info(**args):
    cur_ema_id = args['test_ema_path'].split("/")[-1][:-4]
    spk_path = os.path.join(args['test_ema_path'].split("/")[0], args['test_ema_path'].split("/")[1])
    wav_path = os.path.join(os.path.join(spk_path, 'wav'), cur_ema_id+'.wav')
    wav_data, _ = torchaudio.load(wav_path)
    mel_data = torch.FloatTensor(wav2mel(wav_data)).transpose(0,1).unsqueeze(0)
    etc_path = os.path.join(spk_path, 'etc')
    text_dict_path = os.path.join(etc_path, 'txt.done.data')
    
    emaid2text = {}
    with open(text_dict_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_list = line.split(" ")
            ema_id = line_list[1]
            text = line.split("\"")[1]
            emaid2text[ema_id] = text
    text_trans = emaid2text[cur_ema_id]

    return ema_id, wav_data, mel_data, text_trans

def get_sparsity(H):
    #shape of H is [B, 1, num_gestures, num_points]
    #sparsity = (sqrt(n) - l1/l2) / (sqrt(n) - 1)
    H = H.squeeze(1) #[B, num_gestures, num_points]
    H_l1_c = torch.norm(H, p=1, dim=1) + 1e-6 #[B, num_points]
    H_l2_c = torch.norm(H, p=2, dim=1) + 1e-6 #[B, num_points], plus 1e-6 because H_l2 could be 0 for some vectors
    H_l1_t = torch.norm(H, p=1, dim=2) + 1e-6 #[B, num_gestures]
    H_l2_t = torch.norm(H, p=2, dim=2) + 1e-6 #[B, num_gestures], plus 1e-6 because H_l2 could be 0 for some vectors
    vector_len_c = H.shape[1] #num_gestures
    vector_len_t = H.shape[2] #num_points

    sparsity_c = (math.sqrt(vector_len_c) - H_l1_c/H_l2_c) / (math.sqrt(vector_len_c) - 1)
    sparsity_t = (math.sqrt(vector_len_t) - H_l1_t/H_l2_t) / (math.sqrt(vector_len_t) - 1)
    return sparsity_c, sparsity_t

def vis_H(model, **args):
    ema_id, wav_data, mel_data, text_trans = ema2info(**args)
    ema_data = np.load(args['test_ema_path']) #[t, 12]
    ema_ori, ema_hat, latent_H, _, _ = model(torch.FloatTensor(ema_data).unsqueeze(0).to(device))
    #print(latent_H.shape) #[1,1,num_gestures, t]

    latent_H = latent_H.squeeze().squeeze().detach().numpy() #[num_gesturs, t]
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    im = ax.imshow(latent_H, cmap='hot', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=100)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    #plt.title(text_trans, fontsize=30)
    plt.savefig(os.path.join(args['save_path'], 'latent_H'+"_"+".png"))
    plt.clf()

    #we try to print rows that are "activated"
    _, sparsity_t = get_sparsity(torch.from_numpy(latent_H).unsqueeze(0).unsqueeze(0))
    #print(sparsity_t.shape) #[1, 40] 
    sparse_indices = []
    for i in range(sparsity_t.shape[1]):
        if sparsity_t[0][i] < 1:
            sparse_indices.append(i)
    print("sparse gesture indices", sparse_indices)


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

    ema_id, wav_data, mel_data, text_trans = ema2info(**args)
    
    x = np.arange(ema_data.shape[0])
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(text_trans,fontsize=20)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    outer = gridspec.GridSpec(6, 1, wspace=0.2, hspace=0.2)
    labels = ['tongue dorsum', 'tongue blade', 'tongue tip', 'lower incisor', 'upper lip', 'lower lip']
    for i in range(6):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            _data = ema_data[:,i*2+j] #shape is (win_size,)
            ax.plot(x, _data,c=colors[i], linestyle='dashed', label='ori')
            if mode == 'kinematics':
                ax.plot(x, ema_data_hat[:,i*2+j],c=colors[i], label='rec')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            #ax.get_xaxis().set_ticks([])
            #ax.get_yaxis().set_ticks([])
            if j == 0:
                ax.set_ylabel(labels[i]+' x',rotation=0, fontsize=20, labelpad=10)
            else:
                ax.set_ylabel(labels[i]+' y',rotation=0,fontsize=20, labelpad=10)
            ax.yaxis.set_label_coords(-0.05,0.5)
    plt.savefig(os.path.join(args['save_path'], title+"_"+".png"))
    plt.clf()

def draw_2d(ema_data, ema_data_hat, mode, title, **args):
    
    #fig = plt.figure(figsize=(18, 8))
    #fig.suptitle(title,fontsize=20)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    labels = ['tongue dorsum', 'tongue blade', 'tongue tip', 'lower incisor', 'upper lip', 'lower lip']

    means = []
    stds = []
    stats_path = os.path.join(os.path.join('emadata', 'cin_us_'+args['spk_id']), 'ema.stats')
    with open(stats_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_list = line.split(" ")
            means.append(float(line_list[0]))
            stds.append(float(line_list[1]))
    means = np.array(means)
    stds = np.array(stds)

    stds = 2*np.ones(stds.shape)

    data_x_1 = ema_data[:,0*2] * stds[0] + means[0]
    data_y_1 = ema_data[:,0*2+1] * stds[1] + means[1]
    data_x_2 = ema_data[:,1*2] * stds[2] + means[2]
    data_y_2 = ema_data[:,1*2+1] * stds[3] + means[3]
    data_x_3 = ema_data[:,2*2] * stds[4] + means[4]
    data_y_3 = ema_data[:,2*2+1] * stds[5] + means[5]
    data_x_4 = ema_data[:,3*2] * stds[6] + means[6]
    data_y_4 = ema_data[:,3*2+1] * stds[7] + means[7]
    data_x_5 = ema_data[:,4*2] * stds[8] + means[8]
    data_y_5 = ema_data[:,4*2+1] * stds[9] + means[9]
    data_x_6 = ema_data[:,5*2] * stds[10] + means[10]
    data_y_6 = ema_data[:,5*2+1] * stds[11] + means[11]

    plt.plot(data_x_1, data_y_1, label='tongue dorsum')
    plt.plot(data_x_2, data_y_2, label='tongue blade')
    plt.plot(data_x_3, data_y_3, label='tongue tip')
    plt.plot(data_x_4, data_y_4, label='lower incisor')
    plt.plot(data_x_5, data_y_5, label='upper lip')
    plt.plot(data_x_6, data_y_6, label='lower lip')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.title(title,fontdict = {'fontsize' : 40})
    plt.savefig(os.path.join(args['save_path'], title+"_2d_"+".png"))
    plt.clf()

def vis_gestures(model, **args):
    gestures = model.conv_decoder.weight #[num_pellets, 1, num_gestures, win_size]
    ema_id, wav_data, mel_data, text_trans = ema2info(**args)
    draw_mel(mels=mel_data, mode=ema_id, title=text_trans)
    for i in range(args['num_gestures']):
        gesture_index = i
        #draw_kinematics(gestures[:,0,gesture_index,:].transpose(0,1).detach().numpy(), None, mode='gesture', title='gesture_'+str(gesture_index), **args)
        draw_2d(gestures[:,0,gesture_index,:].transpose(0,1).detach().numpy(), None, mode='gesture', title='gesture_'+str(gesture_index), **args)
    