#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VQ_VAE(nn.Module):
    def __init__(self, commitment_cost=0.25, decay=0.9, ema=False, **args):
        super().__init__()
        
        num_embeddings = args['num_gestures']
        embedding_dim = args['win_size'] * args['num_pellets']
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)

        kmeans_centers = torch.from_numpy(np.load('data/kmeans_pretrain/kmeans_centers_40.npy')) #[40, 12*41=492]
        self._embedding.weight.data = kmeans_centers
        #self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = 1e-5

        self._ema = True

    def forward(self, inputs):

        #shape is B,T,D
        
        # Flatten input
        batch_size = inputs.shape[0]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim) #[B*T, D]
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        #ema
        if self._ema:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

            

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        if self._ema:
            loss_vq = self._commitment_cost * e_latent_loss 
        else:
            loss_vq = q_latent_loss + self._commitment_cost * e_latent_loss
            
        #print(self._embedding.weight)
        
        loss_vq = F.mse_loss(quantized, inputs)
        

        return loss_vq, None, None



class Kmeans_Batch(nn.Module):
    def __init__(self, commitment_cost=1.25, **args):
        super().__init__()
        
        num_embeddings = args['num_gestures']
        embedding_dim = args['win_size'] * args['num_pellets']
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim, requires_grad=False)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):

        #shape is B,T,D
        
        # Flatten input
        batch_size = inputs.shape[0]
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim) #[B*T, D]
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        loss_vq = F.mse_loss(quantized, inputs)

        return loss_vq, None, None