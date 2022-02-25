import sys
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import EMA_Dataset, IEEE_Dataset, rtMRI_Dataset, collate
from models.csnmf import AE_CSNMF, PR_Model, VQ_AE_CSNMF
from models.hifigan_model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from models.utils_hifigan import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from utils import vis_gestures_ema, vis_gestures_rtMRI, vis_kinematics_ema, vis_gestures_ieee, vis_kinematics_ieee, vis_kinematics_rtMRI, vis_H

from trainer import trainer_resynthesis_ema, trainer_resynthesis_ieee, trainer_pr, eval_resynthesis_ema, eval_resynthesis_ieee, eval_pr, trainer_ema2speech

import seaborn as sns
import json
import itertools
from utils import AttrDict, build_env, ema2speech_func
import itertools


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "main python code")
parser.add_argument('--dataset', type=str, default='ema', help='ema, rtMRI')
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
parser.add_argument('--ema2speech', action='store_true', help='')
parser.add_argument('--test_ema2speech', action='store_true', help='')
parser.add_argument('--g2speech', action='store_true', help='')
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
parser.add_argument('--config', type=str, default='', help='')

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
        
    if args.dataset == 'ema':
        args.num_pellets = 12
    elif args.dataset == 'ieee':
        args.num_pellets = 24
    elif args.dataset == 'rtMRI':
        args.num_pellets = 340

    if args.pr_ema or args.pr_mel or args.pr_stft or args.pr_h or args.pr_wav2vec2 or args.pr_mfcc:
        model = PR_Model(**vars(args)).to(device)
    elif args.resynthesis:
        model = AE_CSNMF(**vars(args)).to(device)
    elif args.vq_resynthesis:
        model = VQ_AE_CSNMF(**vars(args)).to(device)
    elif args.ema2speech or args.test_ema2speech:
        with open(args.config) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        generator = Generator(h).to(device)
        mpd = MultiPeriodDiscriminator().to(device)
        msd = MultiScaleDiscriminator().to(device)
        if args.test_ema2speech:
            ema2speech_func(args.test_ema_path, generator)
            print("Finished Generation")
            exit()
            
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
        os.system("sudo rm -rf *.png")
        vis_H(model, **vars(args))
        if args.dataset == 'ema':
            vis_kinematics_ema(model, **vars(args))
            vis_gestures_ema(model, **vars(args))
        elif args.dataset == 'ieee':
            vis_kinematics_ieee(model, **vars(args))
            vis_gestures_ieee(model, **vars(args))  
        elif args.dataset == 'rtMRI':
            vis_kinematics_rtMRI(model, **vars(args))
            vis_gestures_rtMRI(model, **vars(args))  
        exit()
        
    if args.dataset == 'ema':
        dataset_train = EMA_Dataset(mode='train', **vars(args))     
        dataset_test = EMA_Dataset(mode='test', **vars(args))    
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)   
        training_size = len(dataset_train)
        
    elif args.dataset == 'ieee':
        dataset_train = IEEE_Dataset(mode='train', **vars(args))     
        dataset_test = IEEE_Dataset(mode='test', **vars(args))    
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)  
        training_size = len(dataset_train)
        
    elif args.dataset == 'rtMRI':
        dataset_train = rtMRI_Dataset(mode='train', **vars(args))     
        dataset_test = rtMRI_Dataset(mode='test', **vars(args))    
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False)  
        training_size = len(dataset_train)
        
    else:
        print("No dataset specified!!")
        exit()
        
    
    if args.eval_pr:
        print("Eval PER:")
        ctc_loss, per = _eval_pr(model, dataloader_test, device, **vars(args))
        print("PER is: ", per)
        exit()
        
    if args.ema2speech:
        optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
        optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=-1)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=-1)
    else:
        if args.pr_mel or args.pr_mfcc or args.pr_joint or args.pr_stft:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-5, momentum=0.9)
            #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_rate, patience=10, threshold=0.0001)

    if args.pr_mel or args.pr_ema or args.pr_stft or args.pr_wav2vec2 or args.pr_mfcc:
        trainer_pr(model, optimizer, lr_scheduler, dataloader_train, dataloader_test, device, training_size, **vars(args))
    elif args.resynthesis or args.vq_resynthesis:
        if args.dataset == 'ema':
            trainer_resynthesis_ema(model, optimizer, lr_scheduler, dataloader_train, dataloader_test, device, training_size, **vars(args))
        elif args.dataset == 'ieee':
            trainer_resynthesis_ieee(model, optimizer, lr_scheduler, dataloader_train, dataloader_test, device, training_size, **vars(args))
        elif args.dataset == 'rtMRI':
            trainer_resynthesis_ieee(model, optimizer, lr_scheduler, dataloader_train, dataloader_test, device, training_size, **vars(args))
    elif args.ema2speech:
        trainer_ema2speech(generator, mpd, msd, optim_g, optim_d, scheduler_g, scheduler_d, dataloader_train, dataloader_test, device, training_size, **vars(args))
    else:
        print("Error happens! No Training Function Specified!")
        exit()


