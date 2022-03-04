import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
from utils import get_sparsity

import sys
sys.path.append("src/models/")
from vq import *

class PR_Model(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.pr_mel = args['pr_mel']
        self.pr_stft = args['pr_stft']
        self.pr_mfcc = args['pr_mfcc']
        self.pr_wav2vec2 = args['pr_wav2vec2']
        self.pr_ema = args['pr_ema']
        self.num_phns = 43 #if with blank else 42
        self.pr_joint = args['pr_joint']
        
        if self.pr_mel:
            self.in_channels = 80
        elif self.pr_stft:
            self.in_channels = 201
        elif self.pr_mfcc:
            self.in_channels = 39
        elif self.pr_wav2vec2:
            self.in_channels = 201
        elif self.pr_ema:
            self.in_channels = 12
        elif self.pr_joint:
            self.in_channels = args['num_gestures']
        else:
            print("Error!! No ")
            
        self.hidden_size = 256
        self.cnn_encoder1 = nn.Conv1d(in_channels=self.in_channels,out_channels=self.hidden_size // 2,kernel_size=3, padding=1, stride=1, bias=True)
        self.bn1 = nn.BatchNorm1d(self.hidden_size // 2)
        self.elu1 = nn.ELU()
        self.cnn_encoder2 = nn.Conv1d(in_channels=self.hidden_size // 2,out_channels=self.hidden_size,kernel_size=3, padding=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.elu2 = nn.ELU()        

        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size, #256
            hidden_size=self.hidden_size,
            num_layers=3,
            bidirectional=True,
            dropout=0.1
        )

        self.linear_encoder = nn.Linear(2*self.hidden_size, self.num_phns)
        self.init_parameters()
        
    def init_parameters(self):      
        
        #init_cnn
        nn.init.kaiming_normal_(self.cnn_encoder1.weight.data)
        nn.init.normal_(self.cnn_encoder1.bias.data)
        nn.init.kaiming_normal_(self.cnn_encoder2.weight.data)
        nn.init.normal_(self.cnn_encoder2.bias.data)
        
        #init_bn
        self.bn1.weight.data.normal_(1.0, 0.02)
        self.bn2.weight.data.normal_(1.0, 0.02)
        
        #init_lstm
        #from weight_drop import WeightDrop
        #weight_names = [name for name, param in self.lstm.named_parameters() if 'weight' in name]
        #self.lstm = WeightDrop(self.lstm, weight_names, dropout=0.2)
                
        #init_linear
#         nn.init.kaiming_normal_(self.linear_encoder.weight.data)
#         nn.init.normal_(self.linear_encoder.bias.data) 


    def forward(self, inp_utter, inp_utter_len):

        #inp_utter: [B, T, D]

        x = self.cnn_encoder1(inp_utter.permute(0,2,1))  #[B, D, T]
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.cnn_encoder2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = x.permute(2,0,1) #[B, T, D]
        
        #lstm
        packed_x = pack_padded_sequence(x, inp_utter_len.cpu(), enforce_sorted=False) #X: [max_utterance, batch_size, frame_size]
        packed_out = self.lstm_encoder(packed_x)[0]
        out, out_lens = pad_packed_sequence(packed_out) # out: [max_utterance, batch_size, frame_size]

        # Log softmax after output layer is required since`nn.CTCLoss` expects log probabilities.
        #out = self.transformer(out)
        p_out = self.linear_encoder(out).softmax(2)
        log_p_out = self.linear_encoder(out).log_softmax(2)

        return log_p_out, p_out, out_lens
    
    def loadParameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue
            self_state[name].copy_(param)


class AE_CSNMF(nn.Module):
    def __init__(self, **args):
        super().__init__()

        self.win_size = args['win_size']
        self.t = args['segment_len']
        self.num_pellets = args['num_pellets']
        self.num_gestures = args['num_gestures']
        self.conv_encoder1 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=41,padding=20) #larger is better
        self.conv_encoder2 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1) #smaller is better
        self.conv_encoder3 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder4 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder5 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder6 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder7 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder8 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder9 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder10 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_gestures,kernel_size=3,padding=1)
        self.LSTM = nn.LSTM(input_size=self.num_pellets, hidden_size=self.num_pellets//2, num_layers=2, bidirectional=True, batch_first=True)
        self.bn1 = nn.BatchNorm1d(self.num_pellets)
        self.bn2 = nn.BatchNorm1d(self.num_pellets)
        self.bn3 = nn.BatchNorm1d(self.num_pellets)
        self.bn4 = nn.BatchNorm1d(self.num_pellets)
        self.bn5 = nn.BatchNorm1d(self.num_pellets)
        self.bn6 = nn.BatchNorm1d(self.num_pellets)
        self.bn7 = nn.BatchNorm1d(self.num_pellets)
        self.bn8 = nn.BatchNorm1d(self.num_pellets)
        self.bn9 = nn.BatchNorm1d(self.num_pellets)
        self.bn10 = nn.BatchNorm1d(self.num_pellets)
        self.sparse_c_base = args['sparse_c_base']
        self.sparse_t_base = args['sparse_t_base']
        self.pr_joint = args['pr_joint']
        self.fixed_length = args['fixed_length']

        ######Apply weights of k-means to gestures
        if args['dataset'] == 'ema':
            if self.num_gestures == 20:
                kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_20.npy')) #[40, 12*41=492]
            elif self.num_gestures == 40:
                kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_40.npy')) #[40, 12*41=492]
            elif self.num_gestures == 60:
                kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_60.npy')) #[40, 12*41=492]
            elif self.num_gestures == 80:
                kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_80.npy')) #[40, 12*41=492]
            kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 12, 41]
            kmeans_centers = kmeans_centers.permute(1,0,2) #[12,40,41]

            self.conv_decoder_weight = nn.Parameter(kmeans_centers)
            self.gesture_weight = self.conv_decoder_weight
        elif args['dataset'] == 'ieee':
            #self.gesture_weight = nn.Parameter(torch.randn(self.num_pellets, self.num_gestures, self.win_size)) #[24, 40, 41]
            kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_ieee.npy')) #[40, 24*41=984]
            kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 24, 41]
            kmeans_centers = kmeans_centers.permute(1,0,2) #[24, 40,41]
        elif args['dataset'] == 'rtMRI':
            #self.gesture_weight = nn.Parameter(torch.randn(self.num_pellets, self.num_gestures, self.win_size)) #[340, 40, 41]
            kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_rtMRI.npy')) #[40, 340*41=13940]
            kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 340, 41]
            kmeans_centers = kmeans_centers.permute(1,0,2) #[340, 40,41]
            #kmeans_centers = torch.randn(24,40,41)
            #self.conv_decoder_weight = nn.Parameter(kmeans_centers)
            #self.gesture_weight = self.conv_decoder_weight
            
        self.conv_decoder = nn.Conv2d(in_channels=1,out_channels=self.num_pellets,kernel_size=(self.num_gestures,self.win_size),padding=0)

        #torch.nn.init.normal_(self.conv_decoder.weight)
        
        self.conv_decoder.weight.data = kmeans_centers.unsqueeze(1)
        self.conv_decoder.weight.data= self.conv_decoder.weight.data / 100

        if self.pr_joint:
            self.pr_model = PR_Model(**args)

    def forward(self, x, ema_inp_lens):
        #shape of x is [B,t,A]
        #print(x)
        #print("max", torch.max(x))
        #print("min", torch.min(x))
        
        time_steps = x.shape[1]
        x = x.transpose(-1, -2) #[B, A, t]
        H = self.conv_encoder1(x) #[B, C, t]
        H = self.bn1(H)
        H = self.conv_encoder2(H)#[B, C, t]
        H = self.bn2(H)
        H = self.conv_encoder3(H) #[B, C, t]
        H = self.bn3(H)
        H = F.relu(self.conv_encoder4(H)) #[B, C, t]
        
        #print("11", torch.max(self.conv_decoder.weight))
        
        
#         H, _ = self.LSTM(H.transpose(-1,-2))
#         #print("hhjhj", H.shape)
#         H = H.transpose(-1,-2)
        #H = F.relu(self.conv_encoder5(H)) #[B, C, t]
        #H = F.relu(self.conv_encoder6(H)) #[B, C, t]
        #H = F.relu(self.conv_encoder7(H)) #[B, C, t] . #Three encoder layer is the best!
        #H = F.relu(self.conv_encoder8(H)) #[B, C, t]
        #H = F.relu(self.conv_encoder9(H)) #[B, C, t]
        H = F.relu(self.conv_encoder10(H)) #[B, C, t] . #Three encoder layer is the best!
        #print("max", torch.max(H))
        #print("min", torch.min(H))
        if self.pr_joint:
            if not self.fixed_length:
                log_p_out, p_out, out_lens = self.pr_model(H.permute(0,2,1), ema_inp_lens)
            else:
                B = H.shape[0]
                t = H.shape[2]
                inp_utter_lens = torch.ones(B) * t
                log_p_out, p_out, out_lens = self.pr_model(H.permute(0,2,1), inp_utter_lens)
        latent_H = H
        sparsity_c, sparsity_t, entropy_t, entropy_c = get_sparsity(H)
        sparsity_c = sparsity_c.mean() #[B, T] -> [B]
        sparsity_t = sparsity_t.mean() #[B, D] -> [B]
        
        H = H.unsqueeze(1)
        H = H[:,:,:,:time_steps] #The segment length should be the same as input sequence during testing
        H = F.pad(H, pad=(0,self.win_size-1,0,0,0,0,0,0), mode='constant', value=0)
        inp_hat = self.conv_decoder(H).squeeze(dim=-2) #[B, A, t]
        
        #inp_hat = F.conv1d(H, self.gesture_weight.flip(2), padding=20)
        
        loss_vq = torch.Tensor([0]).cuda()

        if self.pr_joint:
            return x, inp_hat, latent_H, sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens, loss_vq
        else:
            return x, inp_hat, latent_H, sparsity_c, sparsity_t, entropy_t, entropy_c, loss_vq

    def loadParameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue
            self_state[name].copy_(param)
            
            
class VQ_AE_CSNMF(nn.Module):
    def __init__(self, **args):
        super().__init__()

        self.win_size = args['win_size']
        self.t = args['segment_len']
        self.num_pellets = args['num_pellets']
        self.num_gestures = args['num_gestures']
        self.conv_encoder1 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=15,padding=7) #larger is better
        self.conv_encoder2 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1) #smaller is better
        self.conv_encoder3 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder4 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder5 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder6 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder7 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_gestures,kernel_size=3,padding=1)
        self.sparse_c_base = args['sparse_c_base']
        self.sparse_t_base = args['sparse_t_base']
        self.pr_joint = args['pr_joint']
        self.fixed_length = args['fixed_length']
        
        ######Apply weights of k-means to gestures
        if self.num_gestures == 20:
            kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_20.npy')) #[40, 12*41=492]
        elif self.num_gestures == 40:
            kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_40.npy')) #[40, 12*41=492]
        elif self.num_gestures == 60:
            kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_60.npy')) #[40, 12*41=492]
        elif self.num_gestures == 80:
            kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_80.npy')) #[40, 12*41=492]
        kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 12, 41]
        kmeans_centers = kmeans_centers.permute(1,0,2) #[12,40,41]

        self.conv_decoder_weight = nn.Parameter(kmeans_centers)
        self.gesture_weight2 = self.conv_decoder_weight

        self.vq_model = VQ_VAE2()
        #self.vq_model2 = VQ_VAE2(**args)

        if self.pr_joint:
            self.pr_model = PR_Model(**args)
            

    def forward(self, x, ema_inp_lens):
        #shape of x is [B,t,A]
        ori_inp = x

        time_steps = x.shape[1]
        x = x.transpose(-1, -2) #[B, A, t]
        H = F.relu(self.conv_encoder1(x)) #[B, C, t]
        H = F.relu(self.conv_encoder2(H)) #[B, C, t]
        #H = F.relu(self.conv_encoder3(H)) #[B, C, t]
        #H = F.relu(self.conv_encoder4(H)) #[B, C, t]
        #H = F.relu(self.conv_encoder5(H)) #[B, C, t]
        #H = F.relu(self.conv_encoder6(H)) #[B, C, t]
        H = F.relu(self.conv_encoder7(H)) #[B, C, t] . #Three encoder layer is the best!
        if self.pr_joint:
            if not self.fixed_length:
                log_p_out, p_out, out_lens = self.pr_model(H.permute(0,2,1), ema_inp_lens)
            else:
                B = H.shape[0]
                t = H.shape[2]
                inp_utter_lens = torch.ones(B) * t
                log_p_out, p_out, out_lens = self.pr_model(H.permute(0,2,1), inp_utter_lens)
        latent_H = H
        sparsity_c, sparsity_t, entropy_t, entropy_c = get_sparsity(H)
        sparsity_c = sparsity_c.mean() #[B, T] -> [B]
        sparsity_t = sparsity_t.mean() #[B, D] -> [B]
        
        x_transpose = x.transpose(-1,-2) #[B, T, A]
        x_pad = F.pad(x, pad=(0,0,(self.win_size-1)//2,(self.win_size-1)//2,0,0), mode='constant', value=0) #[B,T+win,A]
        x_unfold = x_pad.unfold(1, self.win_size,1) #[B, T, A, win]
        x_unfold_reshape = x_unfold.reshape(x_unfold.shape[0], x_unfold.shape[1], x_unfold.shape[2]*x_unfold.shape[3]) #[B,T,A*win]
        
        
        #loss_vq, quan_x_super, encoding_indices = self.vq_model(x_unfold_reshape)   
        
        #let us focus on vanilla super-vector
        ####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #print("0", ori_inp.shape)
        ema_data_huge = ori_inp.reshape(-1, ori_inp.shape[-1]) #[t_super, 12]
        #print("1", ema_data_huge.shape)
        ema_data_huge = ema_data_huge.transpose(0, 1) #[12, t_super]
        #print("2", ema_data_huge.shape)

        ########################################################
        ####################Then we are going to perform kmeans
        ###########win_size = 41
        ###########inp is [12, 716316]
        ###########it should be [12, 716316*41]
        ###########So we pad 20 on both sides
        ema_data_huge_pad = F.pad(ema_data_huge, pad=(20,20,0,0), mode='constant', value=0) #[12, t_super]
        #print("3", ema_data_huge_pad.shape)
        ####################################

        super_ema_data_huge_list = []
        for t in range(ema_data_huge.shape[1]):
            win_ema = ema_data_huge_pad[:,t:t+41] #[12, 41]
            win_ema = win_ema.reshape(-1) #[12*41=492]
            super_ema_data_huge_list.append(win_ema)

        super_ema_data_huge = torch.stack(super_ema_data_huge_list, dim=0) #[t_super, 492]
        #print("4", super_ema_data_huge.shape)
        
        x_super_batch = super_ema_data_huge
        
        ####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        loss_vq = torch.Tensor([0]).cuda()
        for k in range(10):
            loss_vq, _, _ = self.vq_model(x_super_batch) #inp shape should be [B, 12*41]
        
        self.gesture_weight = self.vq_model._embedding.weight.reshape(self.num_gestures, self.num_pellets, self.win_size).permute(1,0,2) #[40, 12, 41] -> (12, 40, 41)
        
        #inp_hat = F.conv1d(H, self.gesture_weight2.flip(2), padding=(self.win_size-1)//2)
        inp_hat = F.conv1d(H, self.gesture_weight.flip(2), padding=20)
        

        if self.pr_joint:
            return x, inp_hat, latent_H, sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens, loss_vq
        else:
            return x, inp_hat, latent_H, sparsity_c, sparsity_t, entropy_t, entropy_c, loss_vq

    def loadParameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue
            self_state[name].copy_(param)


    
