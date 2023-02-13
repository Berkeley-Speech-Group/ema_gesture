import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math


import sys
sys.path.append("src/models/")
sys.path.append("src/")
from utils import get_sparsity
from conformer_encoder import ConformerLayer, PositionalEncoding


class ConcatSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.idim = idim

    def forward(self, x):
        """Subsample x.
        """

        #inp shape is [B, T, idim]
        B = x.shape[0]

        # T mush be oven
        if x.shape[1] % 2 == 1:
            zero_vector = torch.zeros((B, 1, self.idim))
            x = torch.cat((x, zero_vector), dim=1)
        T = x.shape[1]
        x = x.reshape(B, T//2, -1)

        return x

class ConcatUpsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.idim = idim

    def forward(self, x):
        """Subsample x.
        """

        #inp shape is [B, T, idim]
        B = x.shape[0]

        # T mush be oven
        if x.shape[1] % 2 == 1:
            zero_vector = torch.zeros((B, 1, self.idim))
            x = torch.cat((x, zero_vector), dim=1)
        T = x.shape[1]
        x = x.reshape(B, T*2, -1)

        return x


class EMA_Model(nn.Module):
    def __init__(self, **args):
        super().__init__()

        self.win_size = args['win_size']
        self.t = args['segment_len']
        self.num_pellets = args['num_pellets']
        self.num_gestures = args['num_gestures']
        self.sparse_c_base = args['sparse_c_base']
        self.sparse_t_base = args['sparse_t_base']
        self.fixed_length = args['fixed_length']


        #####-----------------------------------------
        #This is for AE Reconstruction
        self.conformer_encoder_layer1 = ConformerLayer(hid_dim=12, n_head=4, filter_size=5, dropout=0.1)
        self.conformer_encoder_layer2 = ConformerLayer(hid_dim=24, n_head=4, filter_size=3, dropout=0.1)
        self.conformer_encoder_layer3 = ConformerLayer(hid_dim=48, n_head=4, filter_size=3, dropout=0.1)
        self.conformer_encoder_layer4 = ConformerLayer(hid_dim=12, n_head=4, filter_size=3, dropout=0.1)

        self.subsampling_layer1 = ConcatSubsampling(idim=12, odim=12, dropout_rate=0.1) #down sampling by 2X       
        self.subsampling_layer2 = ConcatSubsampling(idim=24, odim=12, dropout_rate=0.1) #down sampling by 2X

        self.upsampling_layer1 = ConcatUpsampling(idim=48, odim=12, dropout_rate=0.1) #up sampling by 2X
        self.upsampling_layer2 = ConcatUpsampling(idim=24, odim=12, dropout_rate=0.1) #up sampling by 2X

        self.conformer_decoder_layer1 = ConformerLayer(hid_dim=48, n_head=4, filter_size=5, dropout=0.1)
        self.conformer_decoder_layer2 = ConformerLayer(hid_dim=24, n_head=4, filter_size=5, dropout=0.1)
        self.conformer_decoder_layer3 = ConformerLayer(hid_dim=12, n_head=4, filter_size=5, dropout=0.1)

        #####-----------------------------------------
        #This is for Gestures Modeling (a codebook)
        ######Apply weights of k-means to gestures
        if self.num_gestures == 20:
            kmeans_centers = torch.from_numpy(np.load('/data/jiachenlian/data_nsf/kmeans_pretrain/kmeans_centers_20.npy')) #[40, 12*41=492]
        elif self.num_gestures == 40:
            kmeans_centers = torch.from_numpy(np.load('/data/jiachenlian/data_nsf/kmeans_pretrain/kmeans_centers_40.npy')) #[40, 12*41=492]
        elif self.num_gestures == 60:
            kmeans_centers = torch.from_numpy(np.load('/data/jiachenlian/data_nsf/kmeans_pretrain/kmeans_centers_60.npy')) #[40, 12*41=492]
        elif self.num_gestures == 80:
            kmeans_centers = torch.from_numpy(np.load('/data/jiachenlian/data_nsf/kmeans_pretrain/kmeans_centers_80.npy')) #[40, 12*41=492]
        kmeans_centers = kmeans_centers.reshape(self.num_gestures, self.num_pellets, 41)#[40, 12, 41]

        self.gestures = kmeans_centers.transpose(1,2) # [B'(num_gestures), T_win(window size), 12]
        self.g1 = None
        self.g5 = None

       
        #####-----------------------------------------
        #this is for Gestural Score Modeling
        self.gs_encoder1 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=15,padding=7) #larger is better
        self.gs_encoder2 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1) #smaller is better
        self.gs_encoder3 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_pellets,kernel_size=3,padding=1)
        self.gs_encoder4 = nn.Conv1d(in_channels=self.num_pellets,out_channels=self.num_gestures,kernel_size=3,padding=1)

        self.gs_conformer_encoder_layer1 = ConformerLayer(hid_dim=self.num_gestures, n_head=4, filter_size=5, dropout=0.1)
        self.gs_conformer_encoder_layer2 = ConformerLayer(hid_dim=2*self.num_gestures, n_head=4, filter_size=3, dropout=0.1)
        self.gs_conformer_encoder_layer3 = ConformerLayer(hid_dim=4*self.num_gestures, n_head=4, filter_size=3, dropout=0.1)
        self.gs_conformer_encoder_layer4 = ConformerLayer(hid_dim=self.num_gestures, n_head=4, filter_size=3, dropout=0.1)

        self.gs_subsampling_layer1 = ConcatSubsampling(idim=self.num_gestures, odim=self.num_gestures, dropout_rate=0.1) #down sampling by 2X       
        self.gs_subsampling_layer2 = ConcatSubsampling(idim=2*self.num_gestures, odim=self.num_gestures, dropout_rate=0.1) #down sampling by 2X

        self.gs_upsampling_layer1 = ConcatUpsampling(idim=4*self.num_gestures, odim=self.num_gestures, dropout_rate=0.1) #up sampling by 2X
        self.gs_upsampling_layer2 = ConcatUpsampling(idim=2*self.num_gestures, odim=self.num_gestures, dropout_rate=0.1) #up sampling by 2X

        self.gs_conformer_decoder_layer1 = ConformerLayer(hid_dim=4*self.num_gestures, n_head=4, filter_size=5, dropout=0.1)
        self.gs_conformer_decoder_layer2 = ConformerLayer(hid_dim=2*self.num_gestures, n_head=4, filter_size=5, dropout=0.1)
        self.gs_conformer_decoder_layer3 = ConformerLayer(hid_dim=self.num_gestures, n_head=4, filter_size=5, dropout=0.1)




    def forward(self, x, ema_inp_lens):
        #shape of x is [B,A,T]
        time_steps = x.shape[2]

        x = x.permute(0, 2, 1)                         # [B, T, 12]

        z1 = self.conformer_encoder_layer1(x)          #out: [B,T,12]
        z = self.subsampling_layer1(z1)                #out: [B,T//2,24]
        z2 = self.conformer_encoder_layer2(z)          #out: [B,T//2,24]
        z = self.subsampling_layer2(z2)                #out: [B,T//4,48]
        z3 = self.conformer_encoder_layer3(z)          #out: [B,T//4,48] 

        #bottleneck

        z = self.conformer_decoder_layer1(z3)          #out: [B,T//4,48] 
        z = self.upsampling_layer1(z)                 #out: [B,T//2,24] 
        z4 = self.conformer_decoder_layer2(z)          #out: [B,T//2,24]  
        z = self.upsampling_layer2(z4)                 #out: [B,T,12]  
        z5 = self.conformer_decoder_layer3(z)          #out: [B,T,12] 

        inp_hat = z5
        inp_hat = inp_hat[:, :time_steps, :]


        #####-----------------------------------------
        #Gestures Modeling

        g1 = self.conformer_encoder_layer1(self.gestures)          #out: [B,T,12]
        g = self.subsampling_layer1(g1)                #out: [B,T//2,24]
        g2 = self.conformer_encoder_layer2(g)          #out: [B,T//2,24]
        g = self.subsampling_layer2(g2)                #out: [B,T//4,48]
        g3 = self.conformer_encoder_layer3(g)          #out: [B,T//4,48] 

        g = self.conformer_decoder_layer1(g3)          #out: [B,T//4,48] 
        g = self.upsampling_layer1(g)                 #out: [B,T//2,24] 
        g4 = self.conformer_decoder_layer2(g)          #out: [B,T//2,24]  
        g = self.upsampling_layer2(g4)                 #out: [B,T,12]  
        g5 = self.conformer_decoder_layer3(g)          #out: [B,T,12] 


        self.g1 = g1
        self.g5 = g5


        #####-----------------------------------------
        #Gestural Score Modeling

        #shape is [B, T, num_gestures]
        H = F.relu(self.gs_encoder1(x.transpose(1,2)))        #out: [B, 12, T]
        H = F.relu(self.gs_encoder2(x.transpose(1,2)))        #out: [B, 12, T]
        H = F.relu(self.gs_encoder3(x.transpose(1,2)))        #out: [B, 12, T]
        H = F.relu(self.gs_encoder4(x.transpose(1,2)))        #out: [B, num_gestures, T]

        H = H.transpose(1,2)

        h1 = self.gs_conformer_encoder_layer1(H)          #out: [B,T,num_gestures]
        h = self.gs_subsampling_layer1(h1)                #out: [B,T//2,num_gestures]
        h2 = self.gs_conformer_encoder_layer2(h)          #out: [B,T//2,2*num_gestures]
        h = self.gs_subsampling_layer2(h2)                #out: [B,T//4,4*num_gestures]
        h3 = self.gs_conformer_encoder_layer3(h)          #out: [B,T//4,4*num_gestures] 

        h = self.gs_conformer_decoder_layer1(h3)          #out: [B,T//4,4*num_gestures] 
        h = self.gs_upsampling_layer1(h)                 #out: [B,T//2,2*num_gestures] 
        h4 = self.gs_conformer_decoder_layer2(h)          #out: [B,T//2,2*num_gestures]  
        h = self.gs_upsampling_layer2(h4)                 #out: [B,T,num_gestures]  
        h5 = self.gs_conformer_decoder_layer3(h)          #out: [B,T,num_gestures] 


        #####-----------------------------------------
        #######CSNMF

        # for z1 and h1
        # h1: [B,  num_gestures, T]
        # g1: [12, num_gestures, win_size]
        # z1: [B, 12, T]

        # for z2 and h2 (Not feasible right now)
        # h2: [B,    2*num_gestures, T//2]
        # g2: [12*2, num_gestures,   win_size//2]
        # z2: [B, 12*2, T//2] 

        # for z3 and h3 (Not feasible right now)
        # h3: [B,    4*num_gestures, 4//2]
        # g3: [12*4, num_gestures,   win_size//4]
        # z3: [B, 12*4, T//4]  

        z1_hat = F.conv1d(h1.transpose(1,2), g1.permute(2,0,1).flip(2), padding=(self.win_size-1)//2) #[B, 12, T]
        g5 = g1[:, :g1.shape[1], :]
        # z5_hat = F.conv1d(h5.transpose(1,2), g5.permute(2,0,1).flip(2), padding=(self.win_size-1)//2) #[B, 12, T]
        z5_hat = F.conv1d(h5.transpose(1,2), self.gestures.permute(2,0,1).flip(2), padding=(self.win_size-1)//2) #[B, 12, T]

        #sparsity for h1
        sparsity_c, sparsity_t, entropy_t, entropy_c = get_sparsity(h5.transpose(1,2))
        sparsity_c = sparsity_c.mean() #[B, T] -> [B]
        sparsity_t = sparsity_t.mean() #[B, D] -> [B]


        return x, inp_hat, (h1,h2,h3,h4,h5), (z1,z2,z3,z4,z5), (z1_hat,z5_hat), sparsity_c, sparsity_t, entropy_c, entropy_t

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


if __name__ == '__main__':

    #test conformer layer
    conformer_layer = ConformerLayer(hid_dim=12, n_head=4, filter_size=5, dropout=0.1)
    inp = torch.randn(4,50,12) #[B,T,D]

    print(f"inp shape is {inp.shape}")
    
    out = conformer_layer(inp)

    #test subsampling
    
    subsampling_layer = Conv2dSubsampling(idim=12, odim=12, dropout_rate=0.1)
    out_sub1, _ = subsampling_layer(out, None)

    print(f"out sub shape is {out_sub1.shape}")

    subsampling_layer = Conv2dSubsampling(idim=12, odim=12, dropout_rate=0.1)
    out_sub2, _ = subsampling_layer(out_sub1, None)

    print(f"out sub shape is {out_sub2.shape}")   

    upsampling_layer = Conv2dUpsampling(idim=12, odim=12, dropout_rate=0.1) 
    out_up, _ = upsampling_layer(out_sub2, None)

    print(f"out up shape is {out_up.shape}")

    out_up2, _ = upsampling_layer(out_up, None)

    print(f"out up shape is {out_up2.shape}")

    

    



