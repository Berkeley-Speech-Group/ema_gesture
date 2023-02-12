import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append("../")
sys.path.append("./src/")
sys.path.append("./src/model/")
sys.path.append("../autovc/")
from MHA import MultiHeadedAttention


########### TDNN Transformer definition, ################

class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)

def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.
    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward compatibility.
    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)

class PositionalEncoding(torch.nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = Swish()

    def forward(self, x):
        """Forward funciton."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
    """

    def __init__(self, channels, kernel_size, bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(self, x):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)

class ConformerLayer(nn.Module):

    def __init__(self, hid_dim, n_head, filter_size, dropout):
        super().__init__()
        # FFN
        self.ff_scale = 0.5
        self.feed_forward = PositionwiseFeedForward(hid_dim, 4 * hid_dim, dropout)
        self.norm_ff = nn.LayerNorm(hid_dim)  # for the FNN module
        # MHA
        self.self_attn = MultiHeadedAttention(n_head, hid_dim, dropout=dropout)
        self.norm_mha = nn.LayerNorm(hid_dim)  # for the MHA module
        # CONV
        self.dropout = nn.Dropout(dropout)
        self.conv_module = ConvolutionModule(hid_dim, filter_size)
        self.norm_conv = nn.LayerNorm(hid_dim)  # for the CNN module
        # FFN
        self.feed_forward_macaron = PositionwiseFeedForward(hid_dim, 4 * hid_dim, dropout)
        self.norm_ff_macaron = nn.LayerNorm(hid_dim)

    def forward(self, x):
        # x: [B, T, D]
        #x, pos_emb = x_input[0], x_input[1]

        residual = x
        x = self.norm_ff_macaron(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        # multi-headed self-attention module
        residual = x
        x = self.norm_mha(x)
        x_att, _ = self.self_attn(x, x, x, type='self')
        x = residual + self.dropout(x_att)

        # convolution module
        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv_module(x))

        # feed forward module
        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        return x#, pos_emb

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
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
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
        x = x.unsqueeze(1)  # (b, 1, t, d)
        x = self.conv(x)    # (b, d, t, d)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conformer_Encoder(nn.Module):
    def __init__(self, input_dim, input_ctx, output_dim, tdnn_nhid, tdnn_layers, n_heads=4, dropout=0.1, bn_dim = 0):
        super().__init__()
        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.tdnn_nhid    = tdnn_nhid

        #the number of each tdnn layer params:  
        #filter_size * tdnn_nhid * tdnn_nhid
        # TODO: package layers into a list 
        self.conv_in = Conv2dSubsampling(input_dim, tdnn_nhid, 0.1)
        self.fc_in   = nn.Linear(tdnn_nhid, tdnn_nhid)
        self.bn_in   = nn.BatchNorm1d(tdnn_nhid)
        self.dropout = nn.Dropout(0.1)
        
        
        #self.fc_final   = nn.Linear(tdnn_nhid, tdnn_nhid) 
        self.conformer = nn.ModuleList([ConformerLayer(tdnn_nhid, n_heads, 31, dropout) for i in range(tdnn_layers)])

        self.bn_final = nn.BatchNorm1d(tdnn_nhid)
        self.fc_out = nn.Linear(tdnn_nhid, output_dim)


    def forward(self, x, softmax=True, reset_flag=None, frame_offset=0):
        #x: batch, frame, dim 
        bsz = x.size()[0]
        #x = torch.transpose(x, 0, 1)         
        #x: batch, frame, dim 
        x, _ = self.conv_in(x, None)
        x = self.fc_in(x)
        x = self.dropout(x)
        x = x.contiguous().view(bsz,  -1, self.tdnn_nhid)

        for conformer in self.conformer:
            x = conformer(x) 
        x = self.bn_final(x.contiguous().view(-1, self.tdnn_nhid)) 
        x = self.fc_out(x)

        x = x.contiguous().view(bsz, -1, self.output_dim)
        return x[:, frame_offset:, :], None

    def clean_hidden(self):
        self.hidden = None

    def reset_hidden(self, h, reset_idx):
        if type(h) == tuple:
            return tuple(self.reset_hidden(v, reset_idx) for v in h)
        else:
            if reset_idx.dim() == 0:
                return Variable(h.data)
            else:
                return Variable(h.data.index_fill_(1, reset_idx, 0))


# if __name__ == '__main__':

#     #test conformer layer
#     conformer_layer = ConformerLayer(hid_dim=12, n_head=4, filter_size=5, dropout=0.1)
#     inp = torch.randn(4,100,12) #[B,T,D]
#     out = conformer_layer(inp)
#     print(f"out shape is {out.shape}")

#     #test subsampling
    
#     subsampling_layer = Conv2dSubsampling(idim=12, odim=12, dropout_rate=0.1)
#     out_sub, _ = subsampling_layer(out, None)
#     print(f"out sub shape is {out_sub.shape}")