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
from models.csnmf import CNMF,AE_CSNMF, AE_CNMF, NegativeClipper
from utils import vis_gestures, vis_kinematics, vis_H
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "main python code")
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--num_pellets', type=int, default=12, help='')
parser.add_argument('--num_gestures', type=int, default=40, help='')
parser.add_argument('--win_size', type=int, default=41, help='')
parser.add_argument('--segment_len', type=int, default=500, help='')
parser.add_argument('--num_epochs', type=int, default=100, help='')
parser.add_argument('--learning_rate', type=float, default=0.01, help='')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='')
parser.add_argument('--model_path', type=str, default='', help='')
parser.add_argument('--save_path', type=str, default='save_models/test', help='')
parser.add_argument('--test_ema_path', type=str, default='', help='')
parser.add_argument('--spk_id', type=str, default='all', help='')
parser.add_argument('--step_size', type=int, default=3, help='step_size')
parser.add_argument('--eval_epoch', type=int, default=5, help='eval_epoch')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
parser.add_argument('--lr_decay_rate',type=float, default=0.95, help='lr_decay_rate')
parser.add_argument('--vis_kinematics', action='store_true', help='')
parser.add_argument('--vis_gestures', action='store_true', help='')
parser.add_argument('--sparse_c', action='store_true', help='')
parser.add_argument('--sparse_t', action='store_true', help='')
parser.add_argument('--rec_factor',type=float, default=1, help='')
parser.add_argument('--sparse_c_factor',type=float, default=1e-3, help='')
parser.add_argument('--sparse_t_factor',type=float, default=1e-4, help='')
parser.add_argument('--sparse_c_base',type=float, default=0.95, help='')
parser.add_argument('--sparse_t_base',type=float, default=0.95, help='')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def eval_model(model, ema_dataloader_test):
    print("###################################################")
    print("###########Start EValuating########################")
    print("###################################################")
    rec_loss_e = []
    sparsity_c_e = []
    sparsity_t_e = []
    for i, ema in enumerate(ema_dataloader_test):
        #ema.shape #[batch_size,segment_len,num_pellets]
        ema = ema.to(device)
        model.eval()
        optimizer.zero_grad()
        inp, inp_hat, _, sparsity_c, sparsity_t = model(ema)
        rec_loss = F.l1_loss(inp, inp_hat, reduction='mean')
        rec_loss_e.append(rec_loss.item())
        sparsity_c_e.append(float(sparsity_c))  
        sparsity_t_e.append(float(sparsity_t))  
    print("| Avg RecLoss is %.4f, Sparsity_c is %.4f, Sparsity_t is %.4f" %(sum(rec_loss_e)/len(rec_loss_e), sum(sparsity_c_e)/len(sparsity_c_e), sum(sparsity_t_e)/len(sparsity_t_e)))

def trainer(model, clipper, optimizer, lr_scheduler, ema_dataset_train, ema_dataset_test, **args):

    ema_dataloader_train = torch.utils.data.DataLoader(dataset=ema_dataset_train, batch_size=args['batch_size'], shuffle=True)
    ema_dataloader_test = torch.utils.data.DataLoader(dataset=ema_dataset_test, batch_size=args['batch_size'], shuffle=False)

    #Write into logs
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    log_path = os.path.join(args['save_path'], "logs.txt")
    f = open(log_path, 'w')
    os.chmod(log_path, 755)
    f.write(args['save_path'] + '\n')
    f.write("Process is " + str(os.getppid()))

    writer = SummaryWriter()
    count = 0
    for e in range(args['num_epochs']):
        rec_loss_e = []
        sparsity_c_e = []
        sparsity_t_e = []
        for i, ema in enumerate(ema_dataloader_train):
            #ema.shape #[batch_size,segment_len,num_pellets]
            ema = ema.to(device)
            training_size = len(ema_dataset_train)
            sys.stdout.write("\rTraining Epoch (%d)| Processing (%d/%d)" %(e, i, training_size/args['batch_size']))
            model.train()
            optimizer.zero_grad()
            inp, inp_hat, _,sparsity_c, sparsity_t = model(ema)
            rec_loss = F.l1_loss(inp, inp_hat, reduction='mean')
            loss = args['rec_factor']*rec_loss

            if args['sparse_c']:
                #loss += -args['sparse_c_factor']*sparsity_c
                loss += args['sparse_c_factor']*(sparsity_c-args['sparse_c_base'])**2
            if args['sparse_t']:
                #loss += -args['sparse_t_factor']*sparsity_t
                loss += args['sparse_t_factor']*(sparsity_t-args['sparse_t_base'])**2

            loss.backward()
            optimizer.step()
            #model.conv_decoder.apply(clipper)
            sys.stdout.write(" rec_loss=%.4f, sparsity_c=%.4f, sparsity_t=%.4f " %(rec_loss.item(), sparsity_c, sparsity_t))
            rec_loss_e.append(rec_loss.item())
            sparsity_c_e.append(float(sparsity_c))
            sparsity_t_e.append(float(sparsity_t))
            writer.add_scalar('Rec_Loss_train', rec_loss.item(), count)
            writer.add_scalar('Sparsity_H_c_train', sparsity_c, count)
            writer.add_scalar('Sparsity_H_t_train', sparsity_t, count)
            count += 1
        print("|Epoch: %d Avg RecLoss is %.4f, Sparsity_c is %.4f, Sparsity_t is %.4f" %(e, sum(rec_loss_e)/len(rec_loss_e), sum(sparsity_c_e)/len(sparsity_c_e), sum(sparsity_t_e)/len(sparsity_t_e)))
        
        if (e+1) % args['step_size'] == 0:
            lr_scheduler.step()
        if (e+1) % args['eval_epoch'] == 0:
            ####start evaluation
            eval_model(model, ema_dataloader_test)

        #save the model every 10 epochs
        if (e + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args['save_path'], "best"+str(e)+".pth"))

        #write into log after each epoch
        f.write("***************************************************************************")
        f.write("epoch: %d \n" %(e))
        f.write("Ave loss is %.4f\n" %(sum(rec_loss_e)/len(rec_loss_e)))
        f.write("Sparsity_c is %.4f\n"%(sum(sparsity_c_e)/len(sparsity_c_e)))
        f.write("Sparsity_t is %.4f\n"%(sum(sparsity_t_e)/len(sparsity_t_e)))
        f.write("batch_size is {} \n".format(args['batch_size']))
        f.write("lr = %.4f \n" %(lr_scheduler.get_last_lr()[0]))

        
if __name__ == "__main__":
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    np.random.seed(12)
    random.seed(12)

    ema_dataset_train = EMA_Dataset(mode='train', **vars(args))     
    ema_dataset_test = EMA_Dataset(mode='test', **vars(args))     
    model = AE_CSNMF(**vars(args)).to(device)
    clipper = NegativeClipper()

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
    if args.vis_gestures:
        vis_H(model, **vars(args))
        vis_gestures(model, **vars(args))
        exit()

    #if there is no eval task, start training:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.lr_decay_rate)
    trainer(model, clipper, optimizer, lr_scheduler, ema_dataset_train, ema_dataset_test,  **vars(args))


