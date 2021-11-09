import torch
import torch.nn as nn
from dataloader import EMA_Dataset

ema_dataset = EMA_Dataset()     
ema_dataloader = torch.utils.data.DataLoader(dataset=ema_dataset, batch_size=32, shuffle=True)

a = 0
for i, ema in enumerate(ema_dataloader):
    #print(ema.shape) #[B, 500, 12]