# Gestural Unit Discovery

A Pytorch Implementation of (Varitional) Auto-Encoder Convolutional Non-Negative Matrix Factorization for Gestural Unit Discovery

## Tracking

### Slides

https://docs.google.com/presentation/d/10XbOxXiPCrw9Go2Qp_-aBY7Cp7l6JDoCT87a2W8VhpM/edit?usp=sharing

### Docs (2022 NAACL, Jan 16 2022)

https://docs.google.com/document/d/10n9Oaaqu7THBYDambokOtfEDqT-cnQs_uYpKxhSUh_Q/edit?usp=sharing

## Datasets

First, download EMA data from http://tts.speech.cs.cmu.edu/gopalakr/emadata.tgz, extract it. Check .gitignore for the path of "emadata". Note that we use "nema" data, which is the standardized version of the original data. There are 4579 wav utterances but 4409 ema kinematics data. To handle such mismatch, we only take the ema data that has its corresponding waveform. Actually the first stage(Task 1 of proposal) focuses on gestural unit discovery and only ema data is used.  

Second, convert the standardized ema kinematics data(nema) to npy.

```
python ./src/preprocess/nema_label2npy.py
```

Third, re-normalize the data within [0, 1] because we are performing NMF. (Right now is step is not required)

```
python ./src/preprocess/normalize_ema_npy.py
```

Fourth, run kmeans on ema huge and supervector:

```
python ./src/kmeans.py
```

## Install CTCBeamDecoder:



```
!git clone --recursive https://github.com/parlance/ctcdecode.git
!pip install wget
%cd ctcdecode
!pip install .
%cd ..
```

## Train with Sparse Gestural Scores


```
python src/main.py --sparse_c --sparse_c_factor 0 --spk_id mngu --sparse_t --sparse_t_factor 1 --learning_rate 1e-3 --save_path save_models/sota --batch_size 8
```

## Launch Tensorboard

```
tensorboard --logdir=runs
```


## Test | Visuallize Gestures, Reconstructions and Mel_Spec


```
python src/main.py --vis_gestures --model_path save_models/sota/best169.pth --test_ema_path emadata/cin_us_mngu0/nema/mngu0_s1_0048.npy --spk_id mngu0 --vis_kinematics
```
