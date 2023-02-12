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


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            # torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim // 2 - 1)), odim),
            # pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, 1, t, in_d)
        x = self.conv(x)    # (b, out_d, t//2, in_d//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f)) # (b, t//2, out_d*ind_d//2) -> (b, t//2, out_d)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dUpsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1, odim, 3, 2,0,1),
            # torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim - 1)*2+1+1+2), odim),
            # pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        """
        x = x.unsqueeze(1)  # (b, 1, t, in_d)
        x = self.conv(x)    # (b, out_d, t*2, in_d*2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f)) # (b, t*2, out_d*ind_d*2) -> (b, t*2, out_d)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


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
        self.proj = nn.Linear(idim*2, odim)

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

        # x = self.proj(x)

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
        self.proj = nn.Linear(idim//2, odim)

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

        # x = self.proj(x)

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

        #hidden_dim should be equal to num of pellets
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

    def forward(self, x, ema_inp_lens):
        #shape of x is [B,A,T]
        time_steps = x.shape[2]

        x = x.permute(0, 2, 1)

        z = self.conformer_encoder_layer1(x)          #out: [B,T,12]
        z = self.subsampling_layer1(z)                #out: [B,T//2,24]
        z = self.conformer_encoder_layer2(z)          #out: [B,T//2,24]
        z = self.subsampling_layer2(z)                #out: [B,T//4,48]
        z = self.conformer_encoder_layer3(z)          #out: [B,T//4,48] 

        #bottleneck

        z = self.conformer_decoder_layer1(z)          #out: [B,T//4,48] 
        z = self.upsampling_layer1(z)                 #out: [B,T//2,24] 
        z = self.conformer_decoder_layer2(z)          #out: [B,T//2,24]  
        z = self.upsampling_layer2(z)                 #out: [B,T,12]  
        z = self.conformer_decoder_layer3(z)          #out: [B,T,12] 

        inp_hat = z
        inp_hat = inp_hat[:, :time_steps, :]

        # print(f"max of ema inp is {torch.max(x)}, min of ema inp is {torch.min(x)}")
        # print(f"max of ema hat is {torch.max(inp_hat)}, min of ema hat is {torch.min(inp_hat)}")
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

    

    



