import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utils import get_sparsity

class NegativeClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0, torch.max(w))
            module.weight.data = w

#The Vanilla CNMF model is only able to train the huge kinematics signal
class CNMF(nn.Module):
    def __init__(self, **args):
        super().__init__()
        #shape of X is [1, t, 12]
        self.win_size = args['win_suze']
        self.t = 500
        self.num_pellets = 12
        self.num_gestures = 100
        self.W = nn.Parameter(torch.randn(self.num_pellets, self.win_size, self.num_gestures)*0.01)#[num_pellets, win_size, num_gestures]
        self.H = nn.Parameter(torch.randn(self.num_gestures, self.t)*0.01)#[num_gestures, t]

    def forward(self, inp):
        #inp shape is [B,t,num_pellets]
        H_pad = F.pad(self.H, pad=(self.win_size-1,0,0,0), mode='constant', value=0)
        H_unfold = H_pad.unfold(1,self.t,1)
        H_unfold = H_unfold.transpose(0,1)
        H_unfold = H_unfold[:self.win_size]
        H = torch.flip(H_unfold, dims=[0]) #[win_size, num_gestures, t]
        inp_hat = torch.matmul(self.W.transpose(0,1), H).sum(dim=0) #[num_pellets, t]
        
        return inp, inp_hat

class AE_CNMF(nn.Module):
    def __init__(self,**args):
        super().__init__()

        self.win_size = args['win_size']
        self.t = args['segment_len']
        self.num_pellets = args['num_pellets']
        self.num_gestures = args['num_gestures']
        self.conv_encoder = nn.ConvTranspose2d(in_channels=self.num_pellets,out_channels=1,kernel_size=(self.num_gestures,self.win_size),padding=0)
        self.conv_decoder = nn.Conv2d(in_channels=1,out_channels=self.num_pellets,kernel_size=(self.num_gestures,self.win_size),padding=0)

    def forward(self, x):
        #shape of x is [B,t,num_pellets]
        #print("max of x", torch.max(x))
        #print("min of x", torch.min(x))
        x = x.transpose(-1, -2) #[B, num_pellets, t]
        #print("before unsqueeze, inp_shape",inp.shape)
        inp = x.unsqueeze(-2)
        #print("after unsqueeze, inp_shape",inp.shape)
        H = F.relu(self.conv_encoder(inp)) #[B, 1, num_gestures, num_points]
        #print("after encoder, H_shape", H.shape)
        H = H[:,:,:,:x.shape[2]] #The segment length should be the same as input sequence during testing
        H = F.pad(H, pad=(0,self.win_size-1,0,0,0,0,0,0), mode='constant', value=0)
        inp_hat = self.conv_decoder(H).squeeze(dim=-2) #[B, ]
        return x, inp_hat

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
        self.conv_encoder = nn.ConvTranspose2d(in_channels=self.num_pellets,out_channels=1,kernel_size=(self.num_gestures,self.win_size),padding=0)
        self.conv_decoder = nn.Conv2d(in_channels=1,out_channels=self.num_pellets,kernel_size=(self.num_gestures,self.win_size),padding=0)
        #nn.init.xavier_uniform(self.conv_encoder.weight)
        #nn.init.xavier_uniform(self.conv_decoder.weight)
        #self.conv_encoder.weight.data = F.relu(self.conv_encoder.weight.data)
        

        ########Apply weights of k-means to gestures
        kmeans_centers = torch.from_numpy(np.load('kmeans_centers.npy')) #[40, 12*41=492]
        kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 12, 41]
        kmeans_centers = kmeans_centers.permute(1,0,2).unsqueeze(1)
        self.conv_decoder.weight.data = kmeans_centers #[12, 1, 40, 41]

        ########ALso apply weights of k-means to conv_encoder
        #print(self.conv_encoder.weight.data.shape) #[num_pellets, 1, num_gestures, win_size]
        #self.conv_encoder.weight.data = kmeans_centers

        #self.conv_instancenorm = nn.InstanceNorm1d(self.num_gestures)
        #self.conv_decoder.weight.data = self.conv_instancenorm(self.conv_decoder.weight.data.squeeze(1)) #[A, num_gestures, win_size]
        #self.conv_decoder.weight.data = self.conv_decoder.weight.data.unsqueeze(1) #[A, 1, num_gestures, win_size]
        
        
        #print(self.conv_decoder.weight.data.shape) #[A, 1, num_gestures, win_size]

        #shape of mean should be [A, 1, num_gestures, 1]

        #self.instancenorm = nn.InstanceNorm1d(self.num_gestures)

    def forward(self, x):
        #print("min of x is", torch.min(x))
        #shape of x is [B,t,num_pellets]
        x = x.transpose(-1, -2) #[B, num_pellets, t]
        #print("before unsqueeze, inp_shape",inp.shape)
        inp = x.unsqueeze(-2)
        #print("after unsqueeze, inp_shape",inp.shape)
        H = torch.sigmoid(self.conv_encoder(inp)) #[B, 1, num_gestures, num_points]

        #H = H - torch.min(H)
        # instance stand col
        H = H / torch.sum(H, dim=2, keepdim=True)
        #print("min of H", torch.min(H))
    
        H = H[:,:,:,:x.shape[2]] #The segment length should be the same as input sequence during testing
        sparsity_c, sparsity_t = get_sparsity(H)
        sparsity_c = sparsity_c.mean()
        sparsity_t = sparsity_t.mean()

        latent_H = H ##[1,1,num_gestures, t]
        H = F.pad(H, pad=(0,self.win_size-1,0,0,0,0,0,0), mode='constant', value=0)
        inp_hat = self.conv_decoder(H).squeeze(dim=-2) #[B, ]
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