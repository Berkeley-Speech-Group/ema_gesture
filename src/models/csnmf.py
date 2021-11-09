import torch
import torch.nn as nn

class CSNMF(nn.Module):
    def __init__(self):
        super().__init__()

        #shape of X is [1, t, 12]
        self.T = 10
        self.t = 500
        self.A = 12
        self.num_gestures = 100
        self.W = nn.Parameter(torch.randn(self.A, self.T, self.t)*0.01)#[12, T=10, C=100]
        self.H = nn.Parameter(torch.randn(self.num_gestures, self.t)*0.01)#[C=100, t]

    def forward(self, inp):

        
        return inp
