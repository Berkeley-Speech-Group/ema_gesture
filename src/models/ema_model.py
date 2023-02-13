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

        #This is for Gestures Modeling (a codebook)
        self.gestures = torch.randn(40, 41, 12) # [B', T_win, 12]

        #this is for Gestural Score Modeling



    def forward(self, x, ema_inp_lens):
        #shape of x is [B,A,T]
        time_steps = x.shape[2]

        x = x.permute(0, 2, 1)

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

    

    



