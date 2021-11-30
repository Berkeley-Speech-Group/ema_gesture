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
        # sparsity_c = torch.norm(H, p=1, dim=2)
        # sparsity_t = torch.norm(H, p=1, dim=3)
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
        self.conv_encoder1 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=41,padding=20) #larger is better
        self.conv_encoder2 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1) #smaller is better
        self.conv_encoder3 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder4 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder5 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder6 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.conv_encoder7 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_gestures,kernel_size=3,padding=1)
        self.sparse_c_base = args['sparse_c_base']
        self.sparse_t_base = args['sparse_t_base']

        ######Apply weights of k-means to gestures
        kmeans_centers = torch.from_numpy(np.load('kmeans_centers.npy')) #[40, 12*41=492]
        kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 12, 41]
        kmeans_centers = kmeans_centers.permute(1,0,2) #[12,40,41]

        self.conv_decoder_weight = kmeans_centers.flip(dims=[1])

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
        #H = H / H.sum(dim=1, keepdim=True) #This is Bad for everything

        #sparsity = (sqrt(n) - l1/l2) / (sqrt(n) - 1)
        # l2_norm = torch.norm(H, p=2, dim=1) #[B, t]
        # vec_len = H.shape[1]
        # l1_norm_by_base = (math.sqrt(vec_len) - self.sparse_c_base * (math.sqrt(vec_len) - 1)) * l2_norm #[B, t]
        # l1_norm_by_base = l1_norm_by_base.unsqueeze(1) #[B, 1, t]
        # H_sum = torch.sum(H, dim=1, keepdim=True) #[B, 1, t]
        # s = H + (l1_norm_by_base-H_sum) / vec_len #[B, 1, t]
        # alpha = 0.5
        # s_hat = (1 - alpha) * (l1_norm_by_base) / vec_len + alpha * s
        #print(s_hat.shape)

        latent_H = H
        sparsity_c, sparsity_t = get_sparsity(H)
        #print(sparsity_c.shape) #[B, t]
        #print(sparsity_t.shape) #[B, C]
        sparsity_c = sparsity_c.mean() #[B, T] -> [B]
        sparsity_t = sparsity_t.mean() #[B, D] -> [B]
        inp_hat = F.conv1d(H, self.conv_decoder_weight.flip(2), padding=20)
        inp_hat = F.relu(inp_hat)
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
