import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ctcdecode import CTCBeamDecoder
from config import PHONEME_LIST, PHONEME_LIST_WITH_BLANK, PHONEME_MAP
from utils import voicing_fn, wav2mel
import Levenshtein
from utils_mel import mel_spectrogram
from models.hifigan_model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss

def eval_resynthesis_ieee(model, ema_dataloader_test, device, **args):


    print("###################################################")
    print("###########Start EValuating########################")
    print("###################################################")
    rec_loss_e = []
    sparsity_c_e = []
    sparsity_t_e = []
    loss_vq_e = []
    loss_ctc_e = []
    for i, ema_batch in enumerate(ema_dataloader_test):
        
        ema_batch = ema_batch.to(device)
        model.eval()
        inp, inp_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, loss_vq = model(ema_batch, None)
        rec_loss = F.l1_loss(inp, inp_hat, reduction='mean')
        rec_loss_e.append(rec_loss.item())
        sparsity_c_e.append(float(sparsity_c))  
        sparsity_t_e.append(float(sparsity_t))  


    print("| Avg RecLoss is %.4f, Sparsity_c is %.4f, Sparsity_t is %.4f" %(sum(rec_loss_e)/len(rec_loss_e), sum(sparsity_c_e)/len(sparsity_c_e), sum(sparsity_t_e)/len(sparsity_t_e)))

def eval_resynthesis_ema(model, ema_dataloader_test, device, **args):

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


def trainer_resynthesis_ema(model, optimizer, lr_scheduler, ema_dataloader_train, ema_dataloader_test, device, training_size, **args):

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
            if args['vq_resynthesis']:
                loss += args['vq_factor']*loss_vq.mean()
                
            #loss = 0 * loss_vq

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            #print(model.vq_model._embedding.weight)
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
            eval_resynthesis_ema(model, ema_dataloader_test, device, **args)
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
            
            
def trainer_ema2speech(generator, mpd, msd, optim_g, optim_d, scheduler_g, scheduler_d, dataloader_train, dataloader_test, device, training_size, **args):

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
        loss_d_e = []
        loss_g_e = []
        loss_mel_e = []
        print("PID is {}".format(os.getppid()))
            
        for i, (ema_batch, wav_data_batch, mel_batch) in enumerate(dataloader_train):

            ema_batch = ema_batch.to(device) #[B, 12, T_ema_seg]
            mel_real = mel_batch.to(device) #[B, T_mel_seg, 80]
            wav_real = wav_data_batch.to(device) #[B, T_wav_seg]
            wav_real = wav_real.unsqueeze(1) #[B, 1, T_wav_seg]
            
            
            sys.stdout.write("\rTraining Epoch (%d)| Processing (%d/%d)" %(e, i, training_size/args['batch_size']))
            
            wav_g_hat = generator(ema_batch) #[B, 1, T]
            #print("before", wav_real.shape)
            #print("before", wav_g_hat.shape)
            
            T_min_wav = min(wav_real.shape[-1], wav_g_hat.shape[-1])
            wav_real = wav_real[:,:,:T_min_wav]
            wav_g_hat = wav_g_hat[:,:,:T_min_wav]
            
            #print("after", wav_real.shape)
            #print("after", wav_g_hat.shape)
            wav_g_hat_mel = mel_spectrogram(y=wav_g_hat.cpu().detach().squeeze(1), n_fft=1025, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False).cuda()#[B, 80, T_mel]

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(wav_real, wav_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(wav_real, wav_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            loss_d_e.append(loss_disc_all.item())
            
            torch.nn.utils.clip_grad_norm_(mpd.parameters(), 100)
            torch.nn.utils.clip_grad_norm_(msd.parameters(), 100)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 100)

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            #print("1", mel_real.shape)
            #print("2", wav_g_hat_mel.shape)
            loss_mel = F.l1_loss(mel_real, wav_g_hat_mel.transpose(-1,-2), reduction='mean') * 10
            loss_mel_e.append(loss_mel)

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(wav_real, wav_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(wav_real, wav_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            
            torch.nn.utils.clip_grad_norm_(mpd.parameters(), 100)
            torch.nn.utils.clip_grad_norm_(msd.parameters(), 100)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 100)

            loss_gen_all.backward()
            loss_g_e.append(loss_gen_all.item())
            optim_g.step()       
            

            sys.stdout.write("loss_mel=%.4f, loss_g=%.4f, loss_d=%.4f" %(loss_mel.item(), loss_gen_all.item(), loss_disc_all.item()))
            count += 1
            
        scheduler_g.step()
        scheduler_d.step()
            
        print("|Epoch: %d Ave Loss_mel is %.4f , Avg Loss_G is %.4f, Avg Loss_D is %.4f" %(e, sum(loss_mel_e)/len(loss_mel_e), sum(loss_g_e)/len(loss_g_e), sum(loss_d_e)/len(loss_d_e)))
        
        torch.save(generator.state_dict(), os.path.join(args['save_path'], "generator_best"+".pth"))
        if (e + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(args['save_path'], "generator_best"+str(e)+".pth"))

        #write into log after each epoch
        f.write("***************************************************************************")
        f.write("epoch: %d \n" %(e))
        f.write("Ave loss_g is %.4f\n" %(sum(loss_g_e)/len(loss_g_e)))
        f.write("Ave loss_d is %.4f\n" %(sum(loss_d_e)/len(loss_d_e)))
        
            
def trainer_resynthesis_ieee(model, optimizer, lr_scheduler, ema_dataloader_train, ema_dataloader_test, device, training_size, **args):

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
            
        for i, ema_batch in enumerate(ema_dataloader_train):

            ema_batch = ema_batch.to(device)
            
            sys.stdout.write("\rTraining Epoch (%d)| Processing (%d/%d)" %(e, i, training_size/args['batch_size']))
            model.train()
            optimizer.zero_grad()
            
            inp, inp_hat, _,sparsity_c, sparsity_t, entropy_t, entropy_c, loss_vq = model(ema_batch, None)
            rec_loss = F.l1_loss(inp, inp_hat, reduction='mean')
            loss = args['rec_factor']*rec_loss


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
            if args['vq_resynthesis']:
                loss += args['vq_factor']*loss_vq.mean()
                
            #loss = 0 * loss_vq

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
            #print(model.vq_model._embedding.weight)
            optimizer.step()

            sys.stdout.write(" rec_loss=%.4f, sparsity_c=%.4f, sparsity_t=%.4f, entropy_t=%.4f, entropy_c=%.4f, loss_vq=%.4f \r" %(rec_loss.item(), sparsity_c, sparsity_t, entropy_t, entropy_c, 100*loss_vq.item()))

            rec_loss_e.append(rec_loss.item())
            sparsity_c_e.append(float(sparsity_c))
            sparsity_t_e.append(float(sparsity_t))
            if args['pr_joint']:
                ctc_loss_e.append(float(loss_ctc))
            writer.add_scalar('Rec_Loss_train', rec_loss.item(), count)
            writer.add_scalar('Sparsity_H_c_train', sparsity_c, count)
            writer.add_scalar('Sparsity_H_t_train', sparsity_t, count)
            count += 1

        print("|Epoch: %d Avg RecLoss is %.4f, Sparsity_c is %.4f, Sparsity_t is %.4f" %(e, sum(rec_loss_e)/len(rec_loss_e), sum(sparsity_c_e)/len(sparsity_c_e), sum(sparsity_t_e)/len(sparsity_t_e)))
        
        if (e+1) % args['step_size'] == 0:
            lr_scheduler.step()
        
        if (e+1) % args['eval_epoch'] == 0:
            ####start evaluation
            eval_resynthesis_ieee(model, ema_dataloader_test, device, **args)

            #lr_scheduler.step(rec_loss.item())

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
