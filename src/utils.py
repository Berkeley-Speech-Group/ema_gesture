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
import librosa
import librosa.display
import math
import random
import shutil
import soundfile as sf
from utils_mel import mel_spectrogram
from scipy.interpolate import make_interp_spline, BSpline

def draw_mel0(mels, name):
    #shape of mels should be [B,D=80, T]
    #mels = F.relu(mels)
    mels = mels.transpose(-1, -2)

    for i in range(len(mels)):
        #mel_i = mels[i]
        #np.save("save_models/dsvae2/"+mode+"_mel_"+str(i)+".npy", mel_i.cpu().detach().numpy())
        plt.imshow(mels[i][:,:].transpose(0,1).cpu().detach().numpy())
        plt.savefig(name+"_mel_"+str(i)+".png")
        plt.clf()

def ema2speech_func(ema_path, generator):
    raw_wav_path = ema_path.split("/")[0] + "/" + ema_path.split("/")[1] + "/" + ema_path.split("/")[2] + "/wav/" + ema_path.split("/")[4][:-4]  + ".wav"
    raw_wav, _ = torchaudio.load(raw_wav_path)
    
    ema_data = torch.FloatTensor(np.load(ema_path))#[T_ema, 12]
    ema_data = ema_data.unsqueeze(0).transpose(-1,-2).cuda() #[1,12, T_ema]
    wav_g_hat = generator(ema_data) #[B,T_ema,12]  -> [B, 1, T_wav], B=1
    print("max", torch.max(wav_g_hat))
    print("min", torch.min(wav_g_hat))
    print("wav_gen_shape", wav_g_hat.shape)
    print("wav_raw_shape", raw_wav.shape)
    mel_gen = mel_spectrogram(wav_g_hat.squeeze(0), n_fft=1025, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False).squeeze(0).transpose(-1,-2) #[T_mel, 80]
    mel_raw = mel_spectrogram(raw_wav, n_fft=1025, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False).squeeze(0).transpose(-1,-2) #[T_mel, 80]
    
    draw_mel0(mel_gen.unsqueeze(0).transpose(-1, -2), name='gen')
    draw_mel0(mel_raw.unsqueeze(0).transpose(-1, -2), name='raw')
    
    print(mel_raw.shape)
    print(mel_gen.shape)
    

    #print("Mel delta:", F.l1_loss(mel_raw, mel_gen, reduction='mean'))
    wav_g_hat = wav_g_hat * 10
    sf.write("gen"+".wav", wav_g_hat.cpu().detach().numpy().reshape(-1),16000,'PCM_24')
    sf.write("raw"+".wav", raw_wav.numpy().reshape(-1),16000,'PCM_24')
    
    

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))
        

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def voicing_fn(inp_str):
    #voicing unvoicing
    # p b m -> 0
    # t d n -> 1
    # ch jh -> 2
    # f v -> 3
    # sh zh -> 4
    # k g ng -> 5
    # s z -> 6
    # th dh -> 7
    
    x = inp_str.replace('p','0')
    x = x.replace('b','0')
    x = x.replace('m','0')
    x = x.replace('t','1')
    x = x.replace('d','1')
    x = x.replace('n','1')
    x = x.replace('j','2')
    x = x.replace('c','2')
    x = x.replace('f','3')
    x = x.replace('v','3')    
    x = x.replace('S','4')
    x = x.replace('Z','4')
    x = x.replace('k','5')
    x = x.replace('g','5')
    x = x.replace('G','5')
    x = x.replace('s','6')
    x = x.replace('z','6')
    x = x.replace('T','7')
    x = x.replace('D','7')
    return x

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
    mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
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

def wav2stft(wav):
    #input size of wav should be [1, length]
    stft = torch.stft(wav, n_fft=400, hop_length=160, win_length=400, center=False, window=torch.hann_window(400))
    stft = (stft[:,:,:,0].pow(2)+stft[:,:,:,1].pow(2)).pow(0.5*1) #[1, D, T]
    return stft.squeeze(0).transpose(0,1)

def wav2mfcc(wav):
    #input size of wav should be [1, length]
    mfcc = torchaudio.transforms.MFCC(n_mfcc=39)
    mfcc_data = mfcc(wav).squeeze(0).transpose(0,1)
    return mfcc_data


def draw_mel2(wav=None, mode=None, title=None):
    y, sr = librosa.load(wav) #y:[21969, ]
    librosa.feature.melspectrogram(y=y, sr=sr)
    #plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.feature.melspectrogram(y=y, sr=sr), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    #plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig("save_models/test/"+mode+"_mel_"+".png")
    plt.clf()

def draw_mel(mels, mode, title):
    #shape of mels should be [B,D=80, T]
    #mels = F.relu(mels)
    mels = mels.transpose(-1, -2)

    for i in range(len(mels)):
        # plt.imshow(mels[i][:,:].transpose(0,1).cpu().detach().numpy())
        # plt.xticks(fontsize=30)
        # plt.yticks(fontsize=30)
        # plt.title(title, fontsize=30)
        # plt.savefig("save_models/test/"+mode+"_mel_"+str(i)+".png")
        # plt.clf()

        #plt.figure(figsize=(10, 4))
        librosa.display.specshow(mels[i][:,:].transpose(0,1).cpu().detach().numpy(), y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig("save_models/test/"+mode+"_mel_"+str(i)+".png")
        plt.clf()

def ema2info(**args):
    cur_ema_id = args['test_ema_path'].split("/")[-1][:-4]
    spk_path = os.path.join(os.path.join(args['test_ema_path'].split("/")[0], args['test_ema_path'].split("/")[1]), args['test_ema_path'].split("/")[2])
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

    return ema_id, wav_path, mel_data, text_trans

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

    t_histogram = (1 - sparsity_t + 1e-5) / (1 - sparsity_t + 1e-5).sum(dim=-1, keepdim=True)
    entropy_t = (-t_histogram * torch.log(t_histogram)).mean(dim=-1).mean() #entropy is always non-negative
    #we need to increase E[-plogp] in order to increase diversity, so we just decrease E[plogp] 
    entropy_t = -entropy_t

    c_histogram = (1 - sparsity_c + 1e-5) / (1 - sparsity_c + 1e-5).sum(dim=-1, keepdim=True)
    entropy_c = (-c_histogram * torch.log(c_histogram)).mean(dim=-1).mean() #entropy is always non-negative
    #we need to increase E[-plogp] in order to increase diversity, so we just decrease E[plogp] 
    entropy_c = -entropy_c
    return sparsity_c, sparsity_t, entropy_t, entropy_c
    

def vis_H(model, **args):
    #ema_id, wav_data, mel_data, text_trans = ema2info(**args)
    ema_data = np.load(args['test_ema_path']) #[t, 12]
    ema_data = ema_data[1462:1527,:]
    ema_data = torch.FloatTensor(ema_data).unsqueeze(0).to(device)
    B = ema_data.shape[0]
    T = ema_data.shape[1]
    ema_len_batch = (torch.ones(B) * T).to(device)
    
    if args['pr_joint']:
        ema_ori, ema_hat, latent_H, sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens  = model(ema_data, ema_len_batch)
    else:
        ema_ori, ema_hat, latent_H, _, _, _, _, _ = model(ema_data, None)

    latent_H = latent_H.squeeze().squeeze().cpu().detach().numpy() #[num_gesturs, t]
    
    #latent_H = latent_H[:,:latent_H.shape[1]//6]
    
    plt.figure(figsize=(20,15)) 
    ax = plt.gca()
    im = ax.imshow(latent_H, cmap='Blues', interpolation='nearest', aspect=2)
    divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="2%", pad=-4.7)
#     cbar = plt.colorbar(im, cax=cax)
#     #cbar.ax.tick_params(labelsize=40)
#     cbar.set_ticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.title(text_trans, fontsize=30)
    #plt.savefig(os.path.join(args['save_path'], 'latent_H'+"_"+".png"), bbox_inches='tight')
    plt.savefig('latent_H'+"_"+".png", bbox_inches='tight')
    plt.clf()

    #we try to print rows that are "activated"
    _, sparsity_t, entropy_t,_ = get_sparsity(torch.from_numpy(latent_H).unsqueeze(0).unsqueeze(0))
    #print(sparsity_t.shape) #[1, 40] 
    sparse_indices = []
    for i in range(sparsity_t.shape[1]):
        print(sparsity_t[0][i])
        if sparsity_t[0][i] < 0.80:
            sparse_indices.append(i)
    print("sparse gesture indices", sparse_indices)


def vis_kinematics_ema(model, **args):
    ######################################
    ############The Original Data
    #####################################
    ema_data = np.load(args['test_ema_path']) #[t, 12]
    ema_id = args['test_ema_path'].split("/")[-1][:-4]
    #draw_kinematics(ema_data, mode=ema_id+'_ori', **args)

    ######################################
    ############Reconstruction
    ######################################
    ema_data = torch.FloatTensor(ema_data).unsqueeze(0).to(device)
    B = ema_data.shape[0]
    T = ema_data.shape[1]
    ema_len_batch = (torch.ones(B) * T).to(device)
    
    if args['pr_joint']:
        ema_ori, ema_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens  = model(ema_data, ema_len_batch)
    else:
        ema_ori, ema_hat,_,_,_,_,_ ,_= model(ema_data, None)

    ema_data_hat = ema_hat.transpose(-1, -2).cpu().detach().numpy()
    
    draw_kinematics(ema_data.cpu(), ema_data_hat, mode='kinematics', title=ema_id+'ori_rec', **args) 
    
def vis_kinematics_rtMRI(model, **args):
    ######################################
    ############The Original Data
    #####################################
    ema_data = np.load(args['test_ema_path']) #[t, 340]
    ema_id = args['test_ema_path'].split("/")[-1][:-4]
    #draw_kinematics(ema_data, mode=ema_id+'_ori', **args)

    ######################################
    ############Reconstruction
    ######################################
    ema_data = torch.FloatTensor(ema_data).unsqueeze(0).to(device)
    B = ema_data.shape[0]
    T = ema_data.shape[1]
    ema_len_batch = (torch.ones(B) * T).to(device)
    
    if args['pr_joint']:
        ema_ori, ema_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens  = model(ema_data, ema_len_batch)
    else:
        ema_ori, ema_hat,_,_,_,_,_ ,_= model(ema_data, None)

    ema_data_hat = ema_hat.transpose(-1, -2).cpu().detach().numpy()
    
    draw_kinematics_rtMRI(ema_data.cpu(), ema_data_hat, mode='kinematics', title=ema_id+'ori_rec', **args) 
    
    
def vis_kinematics_ieee(model, **args):
    ######################################
    ############The Original Data
    #####################################
    ema_data = np.load(args['test_ema_path']) #[t, 12]
    ema_id = args['test_ema_path'].split("/")[-1][:-4]
    #draw_kinematics(ema_data, mode=ema_id+'_ori', **args)

    ######################################
    ############Reconstruction
    ######################################
    ema_data = torch.FloatTensor(ema_data).unsqueeze(0).to(device)
    B = ema_data.shape[0]
    T = ema_data.shape[1]
    ema_len_batch = (torch.ones(B) * T).to(device)
    
    if args['pr_joint']:
        ema_ori, ema_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens  = model(ema_data, ema_len_batch)
    else:
        ema_ori, ema_hat,_,_,_,_,_ ,_= model(ema_data, None)

    ema_data_hat = ema_hat.transpose(-1, -2).cpu().detach().numpy()
    
    draw_kinematics_ieee(ema_data.cpu(), ema_data_hat, mode='kinematics', title=ema_id+'ori_rec', **args) 
    

def draw_kinematics_ema(ema_data, ema_data_hat, mode, title, **args):
    

    ema_data = ema_data.squeeze(0)
    if mode == 'kinematics':
        ema_data_hat = ema_data_hat.squeeze(0)
    
    ema_id, wav_data, mel_data, text_trans = ema2info(**args)

    
    x = np.arange(ema_data.shape[0])

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(text_trans,fontsize=20)
    colors = ['red', 'blue', 'black', 'orange', 'purple', 'grey']
    
    outer = gridspec.GridSpec(6, 1, wspace=0.2, hspace=0.2)
    labels = ['tongue dorsum', 'tongue blade', 'tongue tip', 'lower incisor', 'upper lip', 'lower lip']
    for i in range(6):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            _data = ema_data[:,i*2+j] #shape is (win_size,)
            ax.plot(x, _data,c=colors[i], label='ori')
            #ax.plot(x, _data,c=colors[i], label='ori', linewidth=10)
            if mode == 'kinematics':
                ax.plot(x, ema_data_hat[:,i*2+j],c=colors[i], label='rec', linestyle='dashed', linewidth=10)
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
            
    #plt.savefig(os.path.join(args['save_path'], title+"_"+".png"))
    plt.savefig(title+"_"+".png")
    plt.clf()
    
def draw_kinematics_ieee(ema_data, ema_data_hat, mode, title, **args):
    
    ema_data = ema_data.squeeze(0)
    if mode == 'kinematics':
        ema_data_hat = ema_data_hat.squeeze(0)
    
    x = np.arange(ema_data.shape[0])

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("kinematics")
    colors = ['red', 'blue', 'black', 'orange', 'purple', 'grey', 'yellow', 'pink']
    
    outer = gridspec.GridSpec(8, 1, wspace=0.2, hspace=0.2)
    labels = ['TR', 'TB', 'TT', 'UL', 'LL', 'ML','JAW', 'JAWL']
    for i in range(8):
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        for j in range(3):
            ax = plt.Subplot(fig, inner[j])
            _data = ema_data[:,i*3+j] #shape is (win_size,)
            ax.plot(x, _data,c=colors[i], label='ori')
            #ax.plot(x, _data,c=colors[i], label='ori', linewidth=10)
            if mode == 'kinematics':
                ax.plot(x, ema_data_hat[:,i*3+j],c=colors[i], label='rec', linestyle='dashed', linewidth=2)
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
            
    #plt.savefig(os.path.join(args['save_path'], title+"_"+".png"))
    plt.savefig(title+"_"+".png")
    plt.clf()
    
def draw_kinematics_rtMRI(ema_data, ema_data_hat, mode, title, **args):
    
    ema_data = ema_data.squeeze(0)
    if mode == 'kinematics':
        ema_data_hat = ema_data_hat.squeeze(0)
        
    ema_data = ema_data[:1000]
    ema_data_hat = ema_data_hat[:1000]
    
    x = np.arange(ema_data.shape[0])

    fig = plt.figure(figsize=(18, 8))
    fig.suptitle("kinematics")
    number_of_colors = 170
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    
    outer = gridspec.GridSpec(8, 1, wspace=0.2, hspace=0.2)
    for i in range(8):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            _data = ema_data[:,i*2+j] #shape is (win_size,)
            ax.plot(x, _data,c=colors[i], label='ori')
            #ax.plot(x, _data,c=colors[i], label='ori', linewidth=10)
            if mode == 'kinematics':
                ax.plot(x, ema_data_hat[:,i*2+j],c=colors[i], linestyle='dashed', linewidth=2)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            #ax.get_xaxis().set_ticks([])
            #ax.get_yaxis().set_ticks([])
            #if j == 0:
            #    ax.set_ylabel(labels[i]+' x',rotation=0, fontsize=20, labelpad=10)
            #else:
            #    ax.set_ylabel(labels[i]+' y',rotation=0,fontsize=20, labelpad=10)
            #ax.yaxis.set_label_coords(-0.05,0.5)
            
    #plt.savefig(os.path.join(args['save_path'], title+"_"+".png"))
    plt.savefig(title+"_"+".png")
    plt.clf()
    

def draw_2d_ema(ema_data, ema_data_hat, mode, title, **args):
    
    spk_id = 'mngu0'
    
    #fig = plt.figure(figsize=(18, 8))
    #fig.suptitle(title,fontsize=20)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    labels = ['tongue dorsum', 'tongue blade', 'tongue tip', 'lower incisor', 'upper lip', 'lower lip']
    means = []
    stds = []
    stats_path = os.path.join(os.path.join('data/emadata', 'cin_us_'+spk_id), 'ema.stats')
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
    
    data_x = []
    data_y = []
    
    for i in range(6):
        data_x.append(ema_data[:,i*2] * stds[i] + means[i])
        data_y.append(ema_data[:,i*2+1] * stds[i+1] + means[i+1])
        
    indices_new = 2 * np.arange((len(data_x[0]) // 2) + 1)
    
    for i in range(6):
        data_x[i] = data_x[i][indices_new]
        data_y[i] = data_y[i][indices_new]
        
    fig = plt.figure(figsize=(10, 10))

    len_data = len(data_x[0])
    
    number_of_colors = 6

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    
    for i in range(6):
        plt.plot(data_x[i][:len_data//2+5], data_y[i][:len_data//2+5], color=colors[i], linewidth=1)
        plt.plot(data_x[i][len_data//2:], data_y[i][len_data//2:], color=colors[i], linewidth=3)
        
    #plt.xticks(fontsize=50)
    #plt.yticks(fontsize=50)
    plt.xticks([])
    plt.yticks([])
    #plt.legend(prop={'size': 50})
    plt.title(title,fontdict = {'fontsize' : 50})
    #plt.savefig(os.path.join(args['save_path'], title+"_2d_"+".png"))
    plt.savefig(title+"_2d_"+".png")
    plt.clf()
    
    
def draw_2d_rtMRI(ema_data, ema_data_hat, mode, title, **args):
    segment_0 = np.load("data/rtMRI/F_18_Sacramento_sub047/segment0.npy")
    segment_1 = np.load("data/rtMRI/F_18_Sacramento_sub047/segment1.npy")
    segment_2 = np.load("data/rtMRI/F_18_Sacramento_sub047/segment2.npy")
    # print("segment0", segment_0)
    # print("segment1", segment_1)
    # print("segment2", segment_2)
    # print("segment0_shape", segment_0.shape) #[70,1]
    # print("segment1_shape", segment_1.shape) #[40,1]
    # print("segment2_shape", segment_2.shape) #[60,1]
    
    segment_all = []
    prev_segment_label = None
    segment_0 = list(segment_0.reshape(-1))
    segment_1 = list(segment_1.reshape(-1))
    segment_2 = list(segment_2.reshape(-1))
    
    segment_1 = list(len(set(segment_0)) + np.array(segment_1))
    segment_2 = list(len(set(segment_0+segment_1)) + np.array(segment_2))
    
    segment_labs = segment_0 + segment_1 + segment_2
    
    #print("This is draw_2d rtMRI..........")
    
    means = np.load('data/rtMRI/mean.npy').reshape(-1)
    stds = np.load('data/rtMRI/stds.npy').reshape(-1)
    
    data_x = []
    data_y = []
    
    for i in range(170):
        data_x.append(ema_data[:,i*2] * stds[i] + means[i])
        data_y.append(ema_data[:,i*2+1] * stds[i+1] + means[i+1])
        
    indices_new = 2 * np.arange((len(data_x[0]) // 2) + 1)
    
    for i in range(170):
        data_x[i] = data_x[i][indices_new]
        data_y[i] = data_y[i][indices_new]
        
    fig = plt.figure(figsize=(10, 10))

    len_data = len(data_x[0])
    
    number_of_colors = 170

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    
    pre_seg_lab = None
    
    for i in range(170):

        plt.plot(data_x[i][:len_data//2+5], data_y[i][:len_data//2+5], color=colors[segment_labs[i]], linewidth=1)
        plt.plot(data_x[i][len_data//2:], data_y[i][len_data//2:], color=colors[segment_labs[i]], linewidth=3)
        #plt.annotate(str(i), (data_x[i][0], data_y[i][0]))
        cur_seg_lab = segment_labs[i]
        if pre_seg_lab == None:
            pre_seg_lab = cur_seg_lab
        else:
            if cur_seg_lab != pre_seg_lab:
                break
        
    #plt.xticks(fontsize=50)
    #plt.yticks(fontsize=50)
    plt.xticks([])
    plt.yticks([])
    #plt.legend(prop={'size': 50})
    plt.title(title,fontdict = {'fontsize' : 50})
    #plt.savefig(os.path.join(args['save_path'], title+"_2d_"+".png"))
    plt.savefig(title+"_2d_"+".png")
    plt.clf()
    
def draw_new_rtMRI(ema_data, ema_data_hat, mode, title, **args):
    
    
    segment_0 = np.load("data/rtMRI/F_18_Sacramento_sub047/segment0.npy")
    segment_1 = np.load("data/rtMRI/F_18_Sacramento_sub047/segment1.npy")
    segment_2 = np.load("data/rtMRI/F_18_Sacramento_sub047/segment2.npy")
    
    segment_all = []
    prev_segment_label = None
    segment_0 = list(segment_0.reshape(-1))
    segment_1 = list(segment_1.reshape(-1))
    segment_2 = list(segment_2.reshape(-1))
    
    segment_1 = list(len(set(segment_0)) + np.array(segment_1))
    segment_2 = list(len(set(segment_0+segment_1)) + np.array(segment_2))
    
    segment_labs = segment_0 + segment_1 + segment_2
    art_labs = []
    art_labs_unique = []
    for i in range(10):
        art_labs.append("epiglottis")
    art_labs_unique.append("epiglottis")
    for i in range(10, 30):
        art_labs.append("tongue")
    art_labs_unique.append("tongue")
    for i in range(30,35):
        art_labs.append("lower teeth")
    art_labs_unique.append("lower teeth")
    for i in range(35 ,5):
        art_labs.append("lower lip")
    art_labs_unique.append("lower lip")
    for i in range(45,60):
        art_labs.append("chin")
    art_labs_unique.append("chin")
    for i in range(60,70):
        art_labs.append("neck")
    art_labs_unique.append("neck")
    for i in range(70,80):
        art_labs.append("arytenoid")
    art_labs_unique.append("arytenoid")
    for i in range(80,90):
        art_labs.append("pharynx")
    art_labs_unique.append("pharynx")
    for i in range(90,105):
        art_labs.append("back")
    art_labs_unique.append("back")
    for i in range(105, 110):
        art_labs.append("trachea")
    art_labs_unique.append("trachea")
    for i in range(110, 120):
        art_labs.append("hard palate")
    art_labs_unique.append("hard palate")
    for i in range(120, 135):
        art_labs.append("velum")
    art_labs_unique.append("velum")
    for i in range(135,150):
        art_labs.append("nasal cavity")
    art_labs_unique.append("nasal cavity")
    for i in range(150, 160):
        art_labs.append("nose")
    art_labs_unique.append("nose")
    for i in range(160, 170):
        art_labs.append("upper lip")
    art_labs_unique.append("upper lip")
    
    
    #print("This is draw_2d rtMRI..........")
    
    means = np.load('data/rtMRI/mean.npy').reshape(-1)
    stds = np.load('data/rtMRI/stds.npy').reshape(-1)
    
    data_x = []
    data_y = []
    
    for i in range(170):
        data_x.append(ema_data[:,i*2] * stds[2*i] + means[2*i])
        data_y.append(ema_data[:,i*2+1] * stds[2*i+1] + means[2*i+1])
#         data_x.append(ema_data[:,i*2])
#         data_y.append(ema_data[:,i*2+1])
        
#     indices_new = 2 * np.arange((len(data_x[0]) // 2) + 1)
    
#     for i in range(170):
#         data_x[i] = data_x[i][indices_new]
#         data_y[i] = data_y[i][indices_new]
        
    fig = plt.figure(figsize=(10, 10))

    len_data = len(data_x[0])
    
    number_of_colors = 20

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    colors = ['red', 'orange', 'lawngreen', 'lightgreen', 'aqua', 'deepskyblue', 'royalblue', 'blueviolet',\
          'fuchsia', 'yellow', 'crimson', 'thistle', 'cadetblue', 'lightgrey', 'lawngreen', 'slategray', 'peachpuff',\
         'maroon', 'ivory', 'black', 'darkgrey', 'dimgrey']
    
    prev_seg_lab = None
    
    seg_indices = []
    
    for i in range(15):
        seg_indices.append([])
    
    for i in range(170):
        seg_indices[segment_labs[i]-1].append(i)
        
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    for i in range(15):
        seg_ind = np.array(seg_indices[i])
        
#         delta = (args['win_size']-1)//8

#         plt.plot(data_x[seg_ind][:,0], data_y[seg_ind][:,0], color=colors[i], alpha=0.1, linewidth=1)
#         plt.plot(data_x[seg_ind][:,delta], data_y[seg_ind][:,delta], color=colors[i], alpha=0.2, linewidth=1)
#         plt.plot(data_x[seg_ind][:,delta*2], data_y[seg_ind][:,delta*2], color=colors[i], alpha=0.3, linewidth=1)
#         plt.plot(data_x[seg_ind][:,delta*3], data_y[seg_ind][:,delta*3], color=colors[i], alpha=0.4, linewidth=1)
#         plt.plot(data_x[seg_ind][:,delta*4], data_y[seg_ind][:,delta*4], color=colors[i], alpha=0.5, linewidth=1)
#         plt.plot(data_x[seg_ind][:,delta*5], data_y[seg_ind][:,delta*5], color=colors[i], alpha=0.6, linewidth=1)
#         plt.plot(data_x[seg_ind][:,delta*6], data_y[seg_ind][:,delta*6], color=colors[i], alpha=0.7, linewidth=1)
#         plt.plot(data_x[seg_ind][:,delta*7], data_y[seg_ind][:,delta*7], color=colors[i], alpha=0.8, linewidth=1)
#         plt.plot(data_x[seg_ind][:,delta*8], data_y[seg_ind][:,delta*8], color=colors[i], alpha=0.9, linewidth=1, label=art_labs_unique[i])

        delta = (args['win_size']-1)

        plt.plot(data_x[seg_ind][:,0], data_y[seg_ind][:,0], color=colors[i], alpha=0.1, linewidth=1)
        plt.plot(data_x[seg_ind][:,1], data_y[seg_ind][:,1], color=colors[i], alpha=0.2, linewidth=1)
        plt.plot(data_x[seg_ind][:,2], data_y[seg_ind][:,2], color=colors[i], alpha=0.3, linewidth=1)
        plt.plot(data_x[seg_ind][:,3], data_y[seg_ind][:,3], color=colors[i], alpha=0.4, linewidth=1)
        plt.plot(data_x[seg_ind][:,4], data_y[seg_ind][:,4], color=colors[i], alpha=0.5, linewidth=1)
        plt.plot(data_x[seg_ind][:,5], data_y[seg_ind][:,5], color=colors[i], alpha=0.6, linewidth=1)
        plt.plot(data_x[seg_ind][:,6], data_y[seg_ind][:,6], color=colors[i], alpha=0.7, linewidth=1)
        plt.plot(data_x[seg_ind][:,7], data_y[seg_ind][:,7], color=colors[i], alpha=0.8, linewidth=1)
        plt.plot(data_x[seg_ind][:,8], data_y[seg_ind][:,8], color=colors[i], alpha=0.9, linewidth=1, label=art_labs_unique[i])
        


        #break
        #print(data_x[seg_ind][:,0].shape) #[10,]
        #plt.plot(data_x[seg_ind][1], data_y[seg_ind][1], color=colors[i+15])
        #break

        #plt.plot(data_x[i][:len_data//2+5], data_y[i][:len_data//2+5], color=colors[i], linewidth=1)
        #plt.plot(data_x[i][len_data//2:], data_y[i][len_data//2:], color=colors[i], linewidth=3)
        #plt.annotate(str(i), (data_x[i][0], data_y[i][0]))
        
    #plt.xticks(fontsize=50)
    #plt.yticks(fontsize=50)
    plt.xticks([])
    plt.yticks([])
    #plt.legend(prop={'size': 50})
    plt.legend()
    plt.title(title,fontdict = {'fontsize' : 50})
    #plt.savefig(os.path.join(args['save_path'], title+"_2d_"+".png"))
    plt.savefig(title+"_2d_"+".png")
    plt.clf()
    
    
def draw_3d_ieee(ema_data, ema_data_hat, mode, title, **args):
    
    spk_id = 'mngu0'
    
    #fig = plt.figure(figsize=(18, 8))
    #fig.suptitle(title,fontsize=20)
    colors = ['red', 'blue', 'black', 'orange', 'purple', 'grey', 'yellow', 'pink']
    labels = ['TR', 'TB', 'TT', 'UL', 'LL', 'ML','JAW', 'JAWL']
    means = np.load('data/ieee/mean.npy').reshape(-1)
    stds = np.load('data/ieee/stds.npy').reshape(-1)

    data_x = []
    data_y = []
    data_z = []
    for i in range(8):
        data_x.append(ema_data[:,i*3] * stds[i] + means[i])
        data_y.append(ema_data[:,i*3+1] * stds[i+1] + means[i+1])
        data_z.append(ema_data[:,i*3+2] * stds[i+2] + means[i+2])

    indices_new = 2 * np.arange((len(data_x[0]) // 2) + 1)
    
    for i in range(8):
        data_x[i] = data_x[i][indices_new]
        data_y[i] = data_y[i][indices_new]
        data_z[i] = data_z[i][indices_new]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    len_data = len(data_x[i])
    
    for i in range(8):
        ax.plot(data_x[i][:len_data//2+5], data_y[i][:len_data//2+5], data_z[i][:len_data//2+5], color=colors[i], linewidth=1)
        ax.plot(data_x[i][len_data//2:], data_y[i][len_data//2:], data_z[i][len_data//2:], label=labels[i], linewidth=3)
        
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend(prop={'size': 10})
    plt.title(title,fontdict = {'fontsize' : 10})
    plt.savefig(title+"_3d_"+".png")
    plt.clf()
    
    
def vis_gestures_ema(model, **args):
    #gestures = model.conv_decoder.weight #[num_pellets, 1, num_gestures, win_size]
    if args['vq_resynthesis']:
        gestures = model.vq_model._embedding.weight.reshape(args['num_gestures'], args['num_pellets'], args['win_size']).permute(1,0,2).unsqueeze(1)
    else:
        gestures = model.gesture_weight.unsqueeze(1)
    
    ema_id, wav_path, mel_data, text_trans = ema2info(**args)
    draw_mel_ema(mels=mel_data, mode=ema_id, title=text_trans)
    #draw_mel2(wav=wav_path, mode=ema_id, title=text_trans)
    for i in range(args['num_gestures']):
        gesture_index = i
        draw_kinematics_ema(gestures[:,0,gesture_index,:].transpose(0,1).unsqueeze(0).cpu().detach().numpy(), None, mode='gesture', title='Gesture '+str(gesture_index), **args)
        draw_2d_ema(gestures[:,0,gesture_index,:].transpose(0,1).cpu().detach().numpy(), None, mode='gesture', title='Gesture '+str(gesture_index), **args)
    
        
def vis_gestures_ieee(model, **args):
    #gestures = model.conv_decoder.weight #[num_pellets, 1, num_gestures, win_size]
    if args['vq_resynthesis']:
        gestures = model.vq_model._embedding.weight.reshape(args['num_gestures'], args['num_pellets'], args['win_size']).permute(1,0,2).unsqueeze(1)
    else:
        #gestures = model.gesture_weight.unsqueeze(1)
        gestures = 60 * model.conv_decoder.weight
    
    for i in range(args['num_gestures']):
        gesture_index = i
        draw_kinematics_ieee(gestures[:,0,gesture_index,:].transpose(0,1).unsqueeze(0).cpu().detach().numpy(), None, mode='gesture', title='Gesture '+str(gesture_index), **args)
        draw_2d_ieee(gestures[:,0,gesture_index,:].transpose(0,1).cpu().detach().numpy(), None, mode='gesture', title='Gesture '+str(gesture_index), **args)
        

def vis_gestures_rtMRI(model, **args):
    if args['vq_resynthesis']:
        gestures = model.vq_model._embedding.weight.reshape(args['num_gestures'], args['num_pellets'], args['win_size']).permute(1,0,2).unsqueeze(1)
    else:
        gestures = model.conv_decoder.weight
        
    #shape of gestures: [num_pellets, 1, num_gestures, win_size]
    
    
    for i in range(args['num_gestures']):
        gesture_index = i
        #draw_kinematics_rtMRI(gestures[:,0,gesture_index,:].transpose(0,1).unsqueeze(0).cpu().detach().numpy(), None, mode='gesture', title='Gesture '+str(gesture_index), **args)
        draw_new_rtMRI(gestures[:,0,gesture_index,:].transpose(0,1).cpu().detach().numpy(), None, mode='gesture', title='Gesture '+str(gesture_index), **args)
        

    