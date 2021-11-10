import torch
import torch.nn as nn
from dataloader import EMA_Dataset
from models.csnmf import CSNMF,AE_CSNMF

ema_dataset = EMA_Dataset()     
ema_dataloader = torch.utils.data.DataLoader(dataset=ema_dataset, batch_size=16, shuffle=True)
model = AE_CSNMF()

for i, ema in enumerate(ema_dataloader):
    #print(ema.shape) #[1,t=500,12]
    
    model(ema)

