import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataloader import EMA_Dataset
from models.csnmf import CSNMF,AE_CSNMF
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "src code")
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--num_pellets', type=int, default=12, help='')
parser.add_argument('--num_gestures', type=int, default=100, help='')
parser.add_argument('--win_size', type=int, default=10, help='')
parser.add_argument('--segment_len', type=int, default=500, help='')
parser.add_argument('--num_epochs', type=int, default=50, help='')
parser.add_argument('--learning_rate', type=float, default=0.01, help='')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def trainer(model, optimizer):
    writer = SummaryWriter()
    count = 0
    for e in range(args.num_epochs):
        loss_e = []
        for i, ema in enumerate(ema_dataloader):
            #ema.shape #[batch_size,segment_len,num_pellets]
            ema = ema.to(device)
            batch_size = ema.shape[0]
            training_size = len(ema_dataset)
            sys.stdout.write("\rTraining Epoch (%d)| Processing (%d/%d)" %(e, i, training_size/batch_size))
            model.train()
            optimizer.zero_grad()
            ema = ema.to(device)
            inp, inp_hat = model(ema)
            loss = F.l1_loss(inp, inp_hat, reduction='mean')
            loss.backward()
            optimizer.step()
            sys.stdout.write(" loss=%.4f " %(loss.item()))
            loss_e.append(loss.item())
            writer.add_scalar('Loss_train', loss.item(), count)
            count += 1
        print("| Avg Loss in Epoch %d is %.4f" %(e, sum(loss_e)/len(loss_e)))
        

if __name__ == "__main__":
    torch.manual_seed(0)
    ema_dataset = EMA_Dataset(**vars(args))     
    ema_dataloader = torch.utils.data.DataLoader(dataset=ema_dataset, batch_size=args.batch_size, shuffle=True)
    model = AE_CSNMF(**vars(args)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    trainer(model, optimizer)


