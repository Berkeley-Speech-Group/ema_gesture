import torch
import torch.nn as nn
import torch.nn.functional as F

class CSNMF(nn.Module):
    def __init__(self):
        super().__init__()
        #shape of X is [1, t, 12]
        self.win_size = 10
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

class AE_CSNMF(nn.Module):
    def __init__(self):
        super().__init__()

        #shape of X is [1, t, 12]
        self.win_size = 10
        self.t = 500
        self.num_pellets = 12
        self.num_gestures = 100
        self.conv_encoder = nn.ConvTranspose2d(in_channels=self.num_pellets,out_channels=1,kernel_size=(self.num_gestures,self.win_size),padding=0)
        self.conv_decoder = nn.Conv2d(in_channels=1,out_channels=self.num_pellets,kernel_size=(self.num_gestures,self.win_size),padding=0)


    def forward(self, inp):
        #inp shape is [B,t,num_pellets]
        inp = inp.transpose(-1, -2) #[B, num_pellets, t]
        batch_size = inp.shape[0]

        
        #print("before unsqueeze, inp_shape",inp.shape)
        inp = inp.unsqueeze(-2)
        #print("after unsqueeze, inp_shape",inp.shape)

        
        H = self.conv_encoder(inp)
        #print("after encoder, H_shape", H.shape)

        H = H[:,:,:,:self.t]

        H = F.pad(H, pad=(0,self.win_size-1,0,0,0,0,0,0), mode='constant', value=0)
        inp_hat = self.conv_decoder(H).squeeze(dim=-2)
        #print("inp_hat", inp_hat.shape)

        
        return inp, inp_hat
