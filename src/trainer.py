import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ctcdecode import CTCBeamDecoder
from config import PHONEME_LIST, PHONEME_LIST_WITH_BLANK, PHONEME_MAP
from utils import voicing_fn
import Levenshtein

def eval_resynthesis(model, ema_dataloader_test, device, **args):

    if args['pr_joint']:
        criterion = nn.CTCLoss(blank=42, zero_infinity=True)
    print("###################################################")
    print("###########Start EValuating########################")
    print("###################################################")
    rec_loss_e = []
    sparsity_c_e = []
    sparsity_t_e = []
    loss_vq_e = []
    loss_ctc_e = []
    for i, (ema_batch, mel_batch, stft_batch, mfcc_batch, wav2vec2_batch, ema_len_batch, mel_len_batch, stft_len_batch, mfcc_len_batch, wav2vec2_len_batch, lab_batch, lab_len_batch) in enumerate(ema_dataloader_test):
        
        ema_batch = ema_batch.to(device)
        ema_len_batch = ema_len_batch.to(device)
        mel_batch = mel_batch.to(device)
        stft_batch = stft_batch.to(device)
        mfcc_batch = mfcc_batch.to(device)
        wav2vec2_batch = wav2vec2_batch.to(device)
        mel_len_batch = mel_len_batch.to(device)
        stft_len_batch = stft_len_batch.to(device)
        mfcc_len_batch = mfcc_len_batch.to(device)
        wav2vec2_len_batch = wav2vec2_len_batch.to(device)
        lab_batch = lab_batch.to(device)
        lab_len_batch = lab_len_batch.to(device)
        model.eval()
        
        if args['pr_joint']:
            inp, inp_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens, loss_vq  = model(ema_batch, ema_len_batch)
        else:
            inp, inp_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, loss_vq = model(ema_batch, None)
            
        if args['pr_joint']:
            loss_ctc = criterion(log_p_out, lab_batch, out_lens, lab_len_batch)
            
        rec_loss = F.l1_loss(inp, inp_hat, reduction='mean')
        rec_loss_e.append(rec_loss.item())
        sparsity_c_e.append(float(sparsity_c))  
        sparsity_t_e.append(float(sparsity_t))  
        if args['pr_joint']:
            loss_ctc_e.append(loss_ctc.item())
         
    if args['pr_joint']:
        print("| Avg RecLoss is %.4f, Sparsity_c is %.4f, Sparsity_t is %.4f, loss_ctc is %.4f" %(sum(rec_loss_e)/len(rec_loss_e), sum(sparsity_c_e)/len(sparsity_c_e), sum(sparsity_t_e)/len(sparsity_t_e), sum(loss_ctc_e)/len(loss_ctc_e)))
    else:
        print("| Avg RecLoss is %.4f, Sparsity_c is %.4f, Sparsity_t is %.4f" %(sum(rec_loss_e)/len(rec_loss_e), sum(sparsity_c_e)/len(sparsity_c_e), sum(sparsity_t_e)/len(sparsity_t_e)))

def eval_pr(model, ema_dataloader_test, device, **args):
    print("###################################################")
    print("###########Start EValuating(PR)########################")
    print("###################################################")
    criterion = nn.CTCLoss(blank=42, zero_infinity=True)
    ctc_loss_e = []
    edit_distance = 0.0
    count_edit = 0
    for i, (ema_batch, mel_batch, stft_batch, mfcc_batch, wav2vec2_batch, ema_len_batch, mel_len_batch, stft_len_batch, mfcc_len_batch, wav2vec2_len_batch, lab_batch, lab_len_batch) in enumerate(ema_dataloader_test):
        
        ema_batch = ema_batch.to(device)
        ema_len_batch = ema_len_batch.to(device)
        mel_batch = mel_batch.to(device)
        stft_batch = stft_batch.to(device)
        mfcc_batch = mfcc_batch.to(device)
        wav2vec2_batch = wav2vec2_batch.to(device)
        mel_len_batch = mel_len_batch.to(device)
        stft_len_batch = stft_len_batch.to(device)
        mfcc_len_batch = mfcc_len_batch.to(device)
        wav2vec2_len_batch = wav2vec2_len_batch.to(device)
        lab_batch = lab_batch.to(device)
        lab_len_batch = lab_len_batch.to(device)
        model.eval()
        if args['pr_mel']:
            log_p_out, p_out, out_lens = model(mel_batch, mel_len_batch)
        elif args['pr_stft']:
            log_p_out, p_out, out_lens = model(stft_batch, stft_len_batch)  
        elif args['pr_mfcc']:
            log_p_out, p_out, out_lens = model(mfcc_batch, mfcc_len_batch)
        elif args['pr_wav2vec2']:
            log_p_out, p_out, out_lens = model(wav2vec2_batch, wav2vec2_len_batch)
        elif args['pr_ema'] or args['pr_joint']:
            if args['resynthesis'] or args['vq-resynthesis']:
                x, inp_hat, latent_H, sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens, loss_vq = model(ema_batch, ema_len_batch)
            else:
                log_p_out, p_out, out_lens, loss_vq = model(ema_batch, ema_len_batch) 
        else:
            print("Error!! No Training Task Specified!")
            exit()
        loss = criterion(log_p_out, lab_batch, out_lens, lab_len_batch)
        ctc_loss_e.append(loss.item())

        #decoder = CTCBeamDecoder(PHONEME_LIST, beam_width=args['beam_width'])
        decoder = CTCBeamDecoder(PHONEME_MAP, blank_id=42, beam_width=args['beam_width'])
        p_out = torch.transpose(p_out, 1, 0) #[B, T, D] -> [T, B, D]
        #print(out.shape)
        val_out, _, _, val_out_lens = decoder.decode(p_out, out_lens) # shape of res: batch_size, beam_width, max_label_len(beam_width index =0 is the best)
                                                            # shape of res_lens: batch_size, len_labels
            
#         print("val_out.shape", val_out.shape)
#         print("lab shape", lab_batch.shape)
#         print("val_out_lens", val_out_lens)
#         print("label_len", lab_len_batch)
        cur_batch_size = val_out.shape[0]
        
        res_str = ["" for index in range(cur_batch_size)]
        for m in range(cur_batch_size):
            for j in range(val_out_lens[m, 0]):
                res_str[m] += PHONEME_MAP[val_out[m,0,j]]
                
        # label_seq_batch, [batch_size, max_label_len]
        # label_seq_len_batch [batch_size]
        label_str = ["" for index in range(cur_batch_size)]
        for m in range(cur_batch_size):
            for j in range(lab_len_batch[m]):
                label_str[m] += PHONEME_MAP[lab_batch[m][j]]

        for m in range(cur_batch_size):
            src = res_str[m]
            tar = label_str[m]
            if args['pr_voicing']:
                src = voicing_fn(src)
                tar = voicing_fn(tar)
                
            edit_distance += Levenshtein.distance(src, tar)
            count_edit += 1
            
        torch.cuda.empty_cache()
        del ema_batch
        del ema_len_batch
        del mel_batch
        del stft_batch
        del mfcc_batch
        del wav2vec2_batch
        del mel_len_batch 
        del stft_len_batch
        del mfcc_len_batch 
        del wav2vec2_len_batch
        del lab_batch 
        del lab_len_batch 
        
    ctc_loss = sum(ctc_loss_e)/len(ctc_loss_e)

    sys.stdout.write("Eval CTC_loss=%.4f\n" %(ctc_loss))

    print("PER is: ", edit_distance / count_edit)
    return ctc_loss, edit_distance / count_edit



def trainer_resynthesis(model, optimizer, lr_scheduler, ema_dataloader_train, ema_dataloader_test, device, training_size, **args):

    if args['pr_joint']:
        criterion = nn.CTCLoss(blank=42, zero_infinity=True)

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
        print("PID is {}".format(os.getppid()))
        if args['pr_joint']:
            ctc_loss_e = []

        for i, (ema_batch, mel_batch, stft_batch, mfcc_batch, wav2vec2_batch, ema_len_batch, mel_len_batch, stft_len_batch, mfcc_len_batch, wav2vec2_len_batch, lab_batch, lab_len_batch) in enumerate(ema_dataloader_train):

            ema_batch = ema_batch.to(device)
            ema_len_batch = ema_len_batch.to(device)
            mel_batch = mel_batch.to(device)
            stft_batch = stft_batch.to(device)
            mfcc_batch = mfcc_batch.to(device)
            wav2vec2_batch = wav2vec2_batch.to(device)
            mel_len_batch = mel_len_batch.to(device)
            stft_len_batch = stft_len_batch.to(device)
            mfcc_len_batch = mfcc_len_batch.to(device)
            wav2vec2_len_batch = wav2vec2_len_batch.to(device)
            lab_batch = lab_batch.to(device)
            lab_len_batch = lab_len_batch.to(device)
            sys.stdout.write("\rTraining Epoch (%d)| Processing (%d/%d)" %(e, i, training_size/args['batch_size']))
            model.train()
            optimizer.zero_grad()
            
            if args['pr_joint']:
                inp, inp_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, log_p_out, p_out, out_lens, loss_vq = model(ema_batch, ema_len_batch)
            else:
                inp, inp_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, loss_vq = model(ema_batch, None)
            rec_loss = F.l1_loss(inp, inp_hat, reduction='mean')
            loss = args['rec_factor']*rec_loss

            if args['pr_joint']:
                loss_ctc = criterion(log_p_out, lab_batch, out_lens, lab_len_batch)

            if args['sparse_c']:
                #loss += -args['sparse_c_factor']*sparsity_c
                loss += args['sparse_c_factor']*(sparsity_c-args['sparse_c_base'])**2
            if args['sparse_t']:
                #loss += -args['sparse_t_factor']*sparsity_t
                loss += args['sparse_t_factor']*(sparsity_t-args['sparse_t_base'])**2
            if args['entropy_t']:
                #loss += -args['sparse_c_factor']*sparsity_c
                loss += args['entropy_t_factor']*(entropy_t)
            if args['entropy_c']:
                #loss += -args['sparse_c_factor']*sparsity_c
                loss += args['entropy_c_factor']*(entropy_c)
            if args['pr_joint']:
                loss += args['pr_joint_factor']*loss_ctc
            #if args['vq_resynthesis']:
            #    loss += args['vq_factor']*loss_vq
                
            #loss = 0 * loss_vq

            loss.backward()
            print(model.vq_model._embedding.weight)
            optimizer.step()
            if args['pr_joint']:
                sys.stdout.write(" rec_loss=%.4f, sparsity_c=%.4f, sparsity_t=%.4f, entropy_t=%.4f, entropy_c=%.4f, ctc=%.4f, loss_vq=%.4f " %(rec_loss.item(), sparsity_c, sparsity_t, entropy_t, entropy_c, loss_ctc.item(), 100*loss_vq.item()))
            else:
                sys.stdout.write(" rec_loss=%.4f, sparsity_c=%.4f, sparsity_t=%.4f, entropy_t=%.4f, entropy_c=%.4f, loss_vq=%.4f " %(rec_loss.item(), sparsity_c, sparsity_t, entropy_t, entropy_c, 100*loss_vq.item()))

            rec_loss_e.append(rec_loss.item())
            sparsity_c_e.append(float(sparsity_c))
            sparsity_t_e.append(float(sparsity_t))
            if args['pr_joint']:
                ctc_loss_e.append(float(loss_ctc))
            writer.add_scalar('Rec_Loss_train', rec_loss.item(), count)
            writer.add_scalar('Sparsity_H_c_train', sparsity_c, count)
            writer.add_scalar('Sparsity_H_t_train', sparsity_t, count)
            count += 1
        if args['pr_joint']:
            print("|Epoch: %d Avg RecLoss is %.4f, Sparsity_c is %.4f, Sparsity_t is %.4f, CTC_loss is %.4f" %(e, sum(rec_loss_e)/len(rec_loss_e), sum(sparsity_c_e)/len(sparsity_c_e), sum(sparsity_t_e)/len(sparsity_t_e), sum(ctc_loss_e)/len(ctc_loss_e)))
        else:
            print("|Epoch: %d Avg RecLoss is %.4f, Sparsity_c is %.4f, Sparsity_t is %.4f" %(e, sum(rec_loss_e)/len(rec_loss_e), sum(sparsity_c_e)/len(sparsity_c_e), sum(sparsity_t_e)/len(sparsity_t_e)))
        
        #if (e+1) % args['step_size'] == 0:
        #    lr_scheduler.step()
        
        if (e+1) % args['eval_epoch'] == 0:
            ####start evaluation
            eval_resynthesis(model, ema_dataloader_test, device, **args)
            if args['pr_joint']:
                ctc_loss, per = eval_pr(model, ema_dataloader_test, device, **args)
                lr_scheduler.step(ctc_loss)
            else:
                lr_scheduler.step(rec_loss.item())

        torch.save(model.state_dict(), os.path.join(args['save_path'], "best"+".pth"))
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
        if args['pr_joint']:
            f.write("CTC_loss is %.4f\n"%(sum(ctc_loss_e)/len(ctc_loss_e)))

def trainer_pr(model, optimizer, lr_scheduler, ema_dataloader_train, ema_dataloader_test, device, training_size, **args):

    criterion = nn.CTCLoss(blank=42, zero_infinity=True)

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
        print("PID is {}".format(os.getppid()))
        if (e+1) % args['eval_epoch'] == 0:
            ctc_loss, per = eval_pr(model, ema_dataloader_test, device, **args)
            lr_scheduler.step(ctc_loss)

        ctc_loss_e = []
        for i, (ema_batch, mel_batch, stft_batch, mfcc_batch, wav2vec2_batch, ema_len_batch, mel_len_batch, stft_len_batch, mfcc_len_batch, wav2vec2_len_batch, lab_batch, lab_len_batch) in enumerate(ema_dataloader_train):

            ema_batch = ema_batch.to(device)
            ema_len_batch = ema_len_batch.to(device)
            mel_batch = mel_batch.to(device)
            stft_batch = stft_batch.to(device)
            mfcc_batch = mfcc_batch.to(device)
            wav2vec2_batch = wav2vec2_batch.to(device)
            mel_len_batch = mel_len_batch.to(device)
            stft_len_batch = stft_len_batch.to(device)
            mfcc_len_batch = mfcc_len_batch.to(device)
            wav2vec2_len_batch = wav2vec2_len_batch.to(device)
            lab_batch = lab_batch.to(device)
            lab_len_batch = lab_len_batch.to(device)
            #print("max mel_batch is %.4f, min is %.4f" %(torch.max(mel_batch), torch.min(mel_batch)))
            sys.stdout.write("\rTraining Epoch (%d)| Processing (%d/%d)" %(e, i, training_size/args['batch_size']))
            model.train()
            optimizer.zero_grad()
            if args['pr_mel']:
                log_p_out, p_out, out_lens = model(mel_batch, mel_len_batch)
            elif args['pr_stft']:
                log_p_out, p_out, out_lens = model(stft_batch, stft_len_batch)
            elif args['pr_mfcc']:
                log_p_out, p_out, out_lens = model(mfcc_batch, mfcc_len_batch)
            elif args['pr_wav2vec2']:
                log_p_out, p_out, out_lens = model(stft_batch, stft_len_batch)
            elif args['pr_ema']:
                log_p_out, p_out, out_lens = model(ema_batch, ema_len_batch) 
            else:
                print("Error!! No Training Task Specified!")
                exit()

            loss = criterion(log_p_out, lab_batch, out_lens, lab_len_batch)
            ctc_loss_e.append(loss.item())
            loss.backward()
            optimizer.step()

            writer.add_scalar('CTC_Loss_train', loss.item(), count)
            count += 1

            sys.stdout.write("ctc_loss=%.4f" %(loss.item()))
            
            torch.cuda.empty_cache()
            
            del ema_batch
            del ema_len_batch
            del mel_batch
            del stft_batch
            del mfcc_batch
            del wav2vec2_batch
            del mel_len_batch 
            del stft_len_batch
            del mfcc_len_batch 
            del wav2vec2_len_batch
            del lab_batch 
            del lab_len_batch 
            
            #lr_scheduler.step()
        print("|Epoch: %d Avg CTCLoss is %.4f" %(e, sum(ctc_loss_e)/len(ctc_loss_e)))
        
#         if (e+1) % args['step_size'] == 0:
#             lr_scheduler.step()
        #lr_scheduler.step(loss.item())

        torch.save(model.state_dict(), os.path.join(args['save_path'], "best"+".pth"))
        #save the model every 10 epochs
        if (e + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args['save_path'], "best"+str(e)+".pth"))

        #write into log after each epoch
        f.write("***************************************************************************")
        f.write("epoch: %d \n" %(e))
        f.write("Ave loss is %.4f\n" %(sum(ctc_loss_e)/len(ctc_loss_e)))
        f.write("batch_size is {} \n".format(args['batch_size']))
        #f.write("lr = %.4f \n" %(lr_scheduler.get_last_lr()[0]))
