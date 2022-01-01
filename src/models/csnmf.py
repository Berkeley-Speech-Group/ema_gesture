import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
from utils import get_sparsity

import sys
sys.path.append("src/models/")
from vq import VQ_VAE

class PR_Model(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.pr_mel = args['pr_mel']
        self.pr_ema = args['pr_ema']
        self.pr_mel = args['pr_mel']
        #self.num_phns = args['num_phns']
        self.num_phns = 43 #if with blank else 42
        
        if self.pr_mel:
            self.in_channels = 80
        elif self.pr_ema:
            self.in_channels = 12
        elif self.pr_joint:
            self.in_channels = args['num_gestures']
        else:
            print("Error!! No ")

        self.hidden_size = 512
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


class AE_CSNMF(nn.Module):
    def __init__(self, **args):
        super().__init__()

        self.win_size = args['win_size']
        self.t = args['segment_len']
        self.num_pellets = args['num_pellets']
        self.num_gestures = args['num_gestures']
        self.conv_encoder = nn.ConvTranspose2d(in_channels=self.num_pellets,out_channels=1,kernel_size=(self.num_gestures,self.win_size),padding=0)
        self.conv_encoder1 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder2 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder3 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder4 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder5 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder6 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_gestures,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm1d(self.num_pellets)
        self.bn2 = nn.BatchNorm1d(self.num_pellets)
        self.bn3 = nn.BatchNorm1d(self.num_pellets)
        self.bn4 = nn.BatchNorm1d(self.num_pellets)
        self.bn5 = nn.BatchNorm1d(self.num_pellets)
        self.bn6 = nn.BatchNorm1d(self.num_gestures)
        self.conv_decoder = nn.Conv2d(in_channels=1,out_channels=self.num_pellets,kernel_size=(self.num_gestures,self.win_size),padding=0)
        #nn.init.xavier_uniform(self.conv_encoder.weight)
        #nn.init.xavier_uniform(self.conv_decoder.weight)
        #self.conv_encoder.weight.data = F.relu(self.conv_encoder.weight.data)
        
        ######Apply weights of k-means to gestures
        kmeans_centers = torch.from_numpy(np.load('kmeans_centers.npy')) #[40, 12*41=492]
        kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 12, 41]
        kmeans_centers = kmeans_centers.permute(1,0,2).unsqueeze(1)
        self.conv_decoder.weight.data = kmeans_centers #[12, 1, 40, 41]
        
        ########ALso apply weights of k-means to conv_encoder
        #print(self.conv_encoder.weight.data.shape) #[A, 1, num_gestures, win_size]
        #self.conv_encoder.weight.data = torch.pinverse(kmeans_centers).transpose(-1, -2)

        #self.conv_instancenorm = nn.InstanceNorm1d(self.num_gestures)
        #self.conv_decoder.weight.data = self.conv_instancenorm(self.conv_decoder.weight.data.squeeze(1)) #[A, num_gestures, win_size]
        #self.conv_decoder.weight.data = self.conv_decoder.weight.data.unsqueeze(1) #[A, 1, num_gestures, win_size]
        #print(self.conv_decoder.weight.data.shape) #[A, 1, num_gestures, win_size]

        #shape of mean should be [A, 1, num_gestures, 1]
        #self.instancenorm = nn.InstanceNorm1d(self.num_gestures)

    def forward(self, x):
        #shape of x is [B,t,A]
        time_steps = x.shape[1]
        x = x.transpose(-1, -2) #[B, A, t]
        H = F.softplus(self.conv_encoder1(x)) #[B, A, t]
        H = F.softplus(self.conv_encoder2(H)) #[B, A, t]
        H = F.softplus(self.conv_encoder3(H)) #[B, A, t]
        H = F.softplus(self.conv_encoder4(H)) #[B, A, t]
        H = F.softplus(self.conv_encoder6(H)) #[B, C, t]
        H = H.unsqueeze(1) #[B, 1, C, t]
        H = H / H.sum(dim=2, keepdim=True)

        # inp = x.unsqueeze(-2) #[B, A, 1, t]
        # H = F.softplus(self.conv_encoder(inp)) #[B, 1, C, t]
        # print(H.shape)

        sparsity_c, sparsity_t = get_sparsity(H)
        sparsity_c = sparsity_c.mean() #[B, T] -> [B]
        sparsity_t = sparsity_t.mean() #[B, D] -> [B]
        H = H[:,:,:,:time_steps] #The segment length should be the same as input sequence during testing
        latent_H = H ##[1,1,num_gestures, t]
        H = F.pad(H, pad=(0,self.win_size-1,0,0,0,0,0,0), mode='constant', value=0)
        inp_hat = self.conv_decoder(H).squeeze(dim=-2) #[B, A, t]
        return x, inp_hat, latent_H, sparsity_c, sparsity_t

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

class AE_CSNMF2(nn.Module):
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

        ######Apply weights of k-means to gestures
        if self.num_gestures == 20:
            kmeans_centers = torch.from_numpy(np.load('kmeans_centers_20.npy')) #[40, 12*41=492]
        elif self.num_gestures == 40:
            kmeans_centers = torch.from_numpy(np.load('kmeans_centers_40.npy')) #[40, 12*41=492]
        elif self.num_gestures == 60:
            kmeans_centers = torch.from_numpy(np.load('kmeans_centers_60.npy')) #[40, 12*41=492]
        elif self.num_gestures == 80:
            kmeans_centers = torch.from_numpy(np.load('kmeans_centers_80.npy')) #[40, 12*41=492]
        kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 12, 41]
        kmeans_centers = kmeans_centers.permute(1,0,2) #[12,40,41]

        self.conv_decoder_weight = nn.Parameter(kmeans_centers)
        self.gesture_weight = self.conv_decoder_weight

        if self.pr_joint == args['pr_joint']:
            self.pr_model = PR_Model()

    def forward(self, x):
        #shape of x is [B,t,A]
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
            log_p_out, p_out, out_lens = self.pr_model(H)

        latent_H = H
        sparsity_c, sparsity_t, entropy_t, entropy_c = get_sparsity(H)
        sparsity_c = sparsity_c.mean() #[B, T] -> [B]
        sparsity_t = sparsity_t.mean() #[B, D] -> [B]
        inp_hat = F.conv1d(H, self.gesture_weight.flip(2), padding=20)
        #print(self.conv_decoder_weight)

        if self.pr_joint:
            return x, inp_hat, latent_H, sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens  
        else:
            return x, inp_hat, latent_H, sparsity_c, sparsity_t, entropy_t, entropy_c

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


class AE_CSNMF_VQ(nn.Module):
    def __init__(self, **args):
        super().__init__()

        self.win_size = args['win_size']
        self.t = args['segment_len']
        self.num_pellets = args['num_pellets']
        self.num_gestures = args['num_gestures']
        self.h_conv_encoder1 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=15,padding=7) #larger is better
        self.h_conv_encoder2 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1) #smaller is better
        self.h_conv_encoder3 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.h_conv_encoder4 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.h_conv_encoder5 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.h_conv_encoder6 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.h_conv_encoder7 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_gestures,kernel_size=3,padding=1)
        self.sparse_c_base = args['sparse_c_base']
        self.sparse_t_base = args['sparse_t_base']
        self.project = args['project']

        self.vq_model = VQ_VAE(**args)
        self.gesture_weight = self.vq_model._embedding.weight.reshape(self.num_gestures, self.num_pellets, self.win_size) #[40, 12, 41]
        self.gesture_weight = self.gesture_weight.permute(1, 0, 2) #[12, 40, 41]

    def forward(self, x):
        #shape of x is [B,t,A]
        time_steps = x.shape[1]
        x = x.transpose(-1, -2) #[B, A, t]
        H = F.relu(self.h_conv_encoder1(x)) #[B, C, t]
        H = F.relu(self.h_conv_encoder2(H)) #[B, C, t]
        #H = F.relu(self.h_conv_encoder3(H)) #[B, C, t]
        #H = F.relu(self.h_conv_encoder4(H)) #[B, C, t]
        #H = F.relu(self.h_conv_encoder5(H)) #[B, C, t]
        #H = F.relu(self.h_conv_encoder6(H)) #[B, C, t]
        H = F.relu(self.h_conv_encoder7(H)) #[B, C, t] . #Three encoder layer is the best!

        latent_H = H
        sparsity_c, sparsity_t, entropy_t, entropy_c = get_sparsity(H)
        #print(sparsity_c.shape) #[B, t]
        #print(sparsity_t.shape) #[B, C]
        sparsity_c = sparsity_c.mean() #[B, T] -> [B]
        sparsity_t = sparsity_t.mean() #[B, D] -> [B]

        #######################
        ####vqvae and gestures
        x_transpose = x.transpose(-1,-2) #[B, T, A]
        x_pad = F.pad(x, pad=(0,0,(self.win_size-1)//2,(self.win_size-1)//2,0,0), mode='constant', value=0) #[B,T+win,A]
        x_unfold = x_pad.unfold(1, self.win_size,1) #[B, T, A, win]
        x_unfold_reshape = x_unfold.reshape(x_unfold.shape[0], x_unfold.shape[1], x_unfold.shape[2]*x_unfold.shape[3]) #[B,T,A*win]
        loss_vq, quan_x_super, encoding_indices = self.vq_model(x_unfold_reshape)

        #######################
        ####decoder
        #######################

        inp_hat = F.conv1d(H, self.gesture_weight.flip(2), padding=(self.win_size-1)//2)
        #print(self.vq_model._embedding.weight)

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

class AE_CSNMF_VQ_only(nn.Module):
    def __init__(self, **args):
        super().__init__()

        self.win_size = args['win_size']
        self.t = args['segment_len']
        self.num_pellets = args['num_pellets']
        self.num_gestures = args['num_gestures']

        self.vq_model = VQ_VAE(ema=True, **args)
        self.gesture_weight = self.vq_model._embedding.weight.reshape(self.num_gestures, self.num_pellets, self.win_size) #[40, 12, 41]
        self.gesture_weight = self.gesture_weight.permute(1, 0, 2) #[12, 40, 41]

    def forward(self, x):
        #shape of x is [B,t,A]
        time_steps = x.shape[1]
        x = x.transpose(-1, -2) #[B, A, t]

        #######################
        ####vqvae and gestures
        x_transpose = x.transpose(-1,-2) #[B, T, A]
        x_pad = F.pad(x, pad=(0,0,(self.win_size-1)//2,(self.win_size-1)//2,0,0), mode='constant', value=0) #[B,T+win,A]
        x_unfold = x_pad.unfold(1, self.win_size,1) #[B, T, A, win]
        x_unfold_reshape = x_unfold.reshape(x_unfold.shape[0], x_unfold.shape[1], x_unfold.shape[2]*x_unfold.shape[3]) #[B,T,A*win]
        loss_vq, quan_x_super, encoding_indices = self.vq_model(x_unfold_reshape)
        #print(self.vq_model._embedding.weight)
        return loss_vq

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




    
