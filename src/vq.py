#
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQ_VAE(nn.Module):
    def __init__(self, commitment_cost=0.25, **args):
        super().__init__()
        
        num_embeddings = args['num_gestures']
        embedding_dim = args['win_size'] * args['num_pellets']
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
#         if args['zt_norm_mean']:
#             self._embedding.weight.data = self._embedding.weight.data / torch.linalg.norm(self._embedding.weight.data, dim=-1, ord=2, keepdim=True)
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
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss_vq = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.reshape(batch_size, -1)
        
        return loss_vq, quantized.contiguous(), encoding_indices