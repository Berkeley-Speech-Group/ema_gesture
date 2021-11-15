# Gestural Unit Discovery

A Pytorch Implementation of (Varitional) Auto-Encoder Convolutional Non-Negative Matrix Factorization for Gestural Unit Discovery

## Tracking

### Slides

https://docs.google.com/presentation/d/10XbOxXiPCrw9Go2Qp_-aBY7Cp7l6JDoCT87a2W8VhpM/edit?usp=sharing

### Doc

https://docs.google.com/document/d/10n9Oaaqu7THBYDambokOtfEDqT-cnQs_uYpKxhSUh_Q/edit?usp=sharing

## Datasets

First, download EMA data from http://tts.speech.cs.cmu.edu/gopalakr/emadata.tgz, extract it. Check .gitignore for the path of "emadata". Note that we use "nema" data, which is the standardized version of the original data. There are 4579 wav utterances but 4409 ema kinematics data. To handle such mismatch, we only take the ema data that has its corresponding waveform. Actually the first stage(Task 1 of proposal) focuses on gestural unit discovery and only ema data is used.  

Second, convert the standardized ema kinematics data(nema) to npy.

```
python ./src/preprocess/nema2npy.py
```

Third, re-normalize the data within [0, 1] because we are performing NMF.

```
python ./src/preprocess/normalize_ema_npy.py
```

## Arguments

segment_len is number of ema points that is used during training. It is fixed. If sampling rate is 200Hz, it should be 200 for 1 second utterance. 

win_size is the window size in vanilla CSNMF model(in the proposal). It is also one dimension of the 2D kernel in our 2D Conv layer. 

num_gestures is the number of gestures. It is typically "related" or "close to" the number of phonemes(39 in CMU phn dictionary) or acoustic units(300 in HuBERT). 

model_path is the path of the pretrained model. 

save_path is the path that saves the current model. 

test_ema_path is a path of one nma input that is used for testing. 




## Train


```
python ./src/main.py --segment_len 500 --win_size 10 --batch_size 16 --num_gestures 100 --model_path save_models/xxx --save_path save_models/xxx
```

## Train with Sparse Gestural Scores


```
python ./src/main.py --segment_len 500 --win_size 10 --batch_size 16 --num_gestures 100 --model_path save_models/xxx --save_path save_models/xxx --sparse
```

## Launch Tensorboard

```
tensorboard --logdir=runs
```

## Test | Visuallize Kinematics


```
python ./src/main.py --model_path save_models/xxx --test_ema_path xxx --vis_kinematics
```

## Test | Visuallize Gestures


```
python ./src/main.py --model_path save_models/xxx --test_ema_path xxx --vis_gestures
```
