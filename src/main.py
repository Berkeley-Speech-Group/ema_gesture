import sys
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataloader import EMA_Dataset
from models.csnmf import CSNMF,AE_CSNMF, AE_CNMF
from utils import vis_gestures, vis_kinematics

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "main python code")
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--num_pellets', type=int, default=12, help='')
parser.add_argument('--num_gestures', type=int, default=100, help='')
parser.add_argument('--win_size', type=int, default=10, help='')
parser.add_argument('--segment_len', type=int, default=500, help='')
parser.add_argument('--num_epochs', type=int, default=50, help='')
parser.add_argument('--learning_rate', type=float, default=0.01, help='')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='')
parser.add_argument('--model_path', type=str, default='', help='')
parser.add_argument('--save_path', type=str, default='save_models/test', help='')
parser.add_argument('--test_ema_path', type=str, default='', help='')
parser.add_argument('--step_size', type=int, default=5, help='step_size')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
parser.add_argument('--lr_decay_rate',type=float, default=0.8, help='lr_decay_rate')
parser.add_argument('--vis_kinematics', action='store_true', help='')
parser.add_argument('--vis_gestures', action='store_true', help='')
parser.add_argument('--sparse', action='store_true', help='')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

def trainer(model, optimizer, lr_scheduler):

    #Write into logs
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    log_path = os.path.join(args.save_path, "logs.txt")
    f = open(log_path, 'w')
    os.chmod(log_path, 755)
    f.write(args.save_path + '\n')
    f.write("Process is " + str(os.getppid()))

    writer = SummaryWriter()
    count = 0
    for e in range(args.num_epochs):
        loss_e = []
        for i, ema in enumerate(ema_dataloader):
            #ema.shape #[batch_size,segment_len,num_pellets]
            ema = ema.to(device)
            training_size = len(ema_dataset)
            sys.stdout.write("\rTraining Epoch (%d)| Processing (%d/%d)" %(e, i, training_size/args.batch_size))
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
        lr_scheduler.step()

        #save the model every 10 epochs
        if (e + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, "best"+str(e)+".pth"))

        #write into log after each epoch
        f.write("***************************************************************************")
        f.write("epoch: %d \n" %(e))
        f.write("Ave loss is %.4f\n" %(sum(loss_e)/len(loss_e)))
        f.write("batch_size is {} \n".format(args.batch_size))
        f.write("lr = %.4f \n" %(lr_scheduler.get_last_lr()[0]))

        

if __name__ == "__main__":
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    np.random.seed(12)
    random.seed(12)

    ema_dataset = EMA_Dataset(**vars(args))     
    ema_dataloader = torch.utils.data.DataLoader(dataset=ema_dataset, batch_size=args.batch_size, shuffle=True)
    model = AE_CNMF(**vars(args)).to(device)

    if not os.path.exists(args.model_path):
        print("Model not exist and we just create the new model......")
    else:
        print("Model Exists")
        print("Model Path is " + args.model_path)
        if args.model_path.endswith(".model"):
            model_c = torch.load(args.model_path).cuda() 
            for name, param in model_c.named_parameters():
                if name.__contains__("module"):
                    name = name.replace("module.", "")
                if name in model.state_dict():
                    if model.state_dict()[name].size() != param.size():
                        print("Wrong parameter length: %s, model: %s, loaded: %s"%(name, model.state_dict()[name].size(), param.size()))
                        continue
                    model.state_dict()[name].copy_(param) 
                else:
                    print(name + " is not existed!")
        else:
            model.loadParameters(args.model_path)

    if args.vis_kinematics:
        vis_kinematics(model, **vars(args))
        exit()
    if args.vis_gestures:
        exit()

    #if there is no eval task, start training:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.lr_decay_rate)
    trainer(model, optimizer, lr_scheduler)


