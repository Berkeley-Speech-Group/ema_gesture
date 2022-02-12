import sys
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchnmf.nmf import NMF2D, NMFD
# from torchnmf.metrics import kl_div

from dataloader import EMA_Dataset, collate
from models.csnmf import AE_CSNMF, PR_Model, VQ_AE_CSNMF
from trainer import trainer_resynthesis, trainer_pr, eval_resynthesis, eval_pr
from utils import vis_gestures, vis_kinematics, vis_H
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "main python code")
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--num_pellets', type=int, default=12, help='')
parser.add_argument('--num_gestures', type=int, default=40, help='')
parser.add_argument('--win_size', type=int, default=41, help='')
parser.add_argument('--segment_len', type=int, default=100, help='')
parser.add_argument('--num_epochs', type=int, default=500, help='')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='')
parser.add_argument('--model_path', type=str, default='', help='')
parser.add_argument('--save_path', type=str, default='save_models/test', help='')
parser.add_argument('--test_ema_path', type=str, default='', help='')
parser.add_argument('--spk_id', type=str, default='all', help='')
parser.add_argument('--step_size', type=int, default=3, help='step_size')
parser.add_argument('--eval_epoch', type=int, default=1, help='eval_epoch')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
parser.add_argument('--num_phns', type=int, default=42, help='num_phns')
parser.add_argument('--beam_width', type=int, default=100, help='beam_width')
parser.add_argument('--lr_decay_rate',type=float, default=0.9, help='lr_decay_rate')
parser.add_argument('--vis_kinematics', action='store_true', help='')
parser.add_argument('--vis_gestures', action='store_true', help='')
parser.add_argument('--sparse_c', action='store_true', help='')
parser.add_argument('--sparse_t', action='store_true', help='')
parser.add_argument('--entropy_t', action='store_true', help='')
parser.add_argument('--entropy_c', action='store_true', help='')
parser.add_argument('--rec_factor',type=float, default=1, help='')
parser.add_argument('--sparse_c_factor',type=float, default=1e-3, help='')
parser.add_argument('--sparse_t_factor',type=float, default=1e-4, help='')
parser.add_argument('--entropy_t_factor',type=float, default=1, help='')
parser.add_argument('--entropy_c_factor',type=float, default=1, help='')
parser.add_argument('--vq_factor',type=float, default=10, help='')
parser.add_argument('--ctc_factor',type=float, default=10, help='')
parser.add_argument('--sparse_c_base',type=float, default=0.95, help='')
parser.add_argument('--sparse_t_base',type=float, default=0.95, help='')
parser.add_argument('--NMFD', action='store_true', help='')
parser.add_argument('--project', action='store_true', help='')
parser.add_argument('--with_phoneme', action='store_true', help='')
parser.add_argument('--fixed_length', action='store_true', help='')
parser.add_argument('--pr_mel', action='store_true', help='')
parser.add_argument('--pr_stft', action='store_true', help='')
parser.add_argument('--pr_mfcc', action='store_true', help='')
parser.add_argument('--pr_wav2vec2', action='store_true', help='')
parser.add_argument('--pr_ema', action='store_true', help='')
parser.add_argument('--pr_h', action='store_true', help='')
parser.add_argument('--pr_joint', action='store_true', help='')
parser.add_argument('--pr_joint_factor', type=float, default=1, help='')
parser.add_argument('--asr_wav', action='store_true', help='')
parser.add_argument('--asr_ema', action='store_true', help='')
parser.add_argument('--asr_h', action='store_true', help='')
parser.add_argument('--resynthesis', action='store_true', help='')
parser.add_argument('--vq_resynthesis', action='store_true', help='')
parser.add_argument('--vq', action='store_true', help='')
parser.add_argument('--vq_only', action='store_true', help='to test the clustering performance')
parser.add_argument('--eval_pr', action='store_true', help='')
parser.add_argument('--pr_voicing', action='store_true', help='')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    np.random.seed(12)
    random.seed(12)

    if args.fixed_length:
        print("Fixed Length")
    else:
        print("No Fixed Length")

    if args.pr_ema or args.pr_mel or args.pr_stft or args.pr_h or args.pr_wav2vec2 or args.pr_mfcc:
        model = PR_Model(**vars(args)).to(device)
    elif args.resynthesis:
        model = AE_CSNMF(**vars(args)).to(device)
    elif args.vq_resynthesis:
        model = VQ_AE_CSNMF(**vars(args)).to(device)
    else:
        print("Error!!! No Model Specified!!")
        exit()

    if not os.path.exists(args.model_path):
        print("Model not exist and we just create the new model......")
    else:
        print("Model Exists and Loading........")
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

    if args.vis_gestures:
        print("###################################Visualize Gestures#########################################")
        vis_H(model, **vars(args))
        vis_kinematics(model, **vars(args))
        vis_gestures(model, **vars(args))
        exit()
        
    ema_dataset_train = EMA_Dataset(mode='train', **vars(args))     
    ema_dataset_test = EMA_Dataset(mode='test', **vars(args))    
    training_size = len(ema_dataset_train)
        
    ema_dataloader_train = torch.utils.data.DataLoader(dataset=ema_dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    ema_dataloader_test = torch.utils.data.DataLoader(dataset=ema_dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)    
    
    if args.eval_pr:
        print("Eval PER:")
        ctc_loss, per = _eval_pr(model, ema_dataloader_test, device, **vars(args))
        print("PER is: ", per)
        exit()

    if args.pr_mel or args.pr_mfcc or args.pr_joint or args.pr_stft:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-5, momentum=0.9)
    
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.8)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_rate, patience=4, threshold=0.0001)

    if args.pr_mel or args.pr_ema or args.pr_stft or args.pr_wav2vec2 or args.pr_mfcc:
        trainer_pr(model, optimizer, lr_scheduler, ema_dataloader_train, ema_dataloader_test, device, training_size, **vars(args))
    elif args.resynthesis or args.vq_resynthesis:
        trainer_resynthesis(model, optimizer, lr_scheduler, ema_dataloader_train, ema_dataloader_test, device, training_size, **vars(args))
    else:
        print("Error happens! No Training Function Specified!")
        exit()


