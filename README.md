# Gestural Unit Discovery

A Pytorch Implementation of (Varitional) Auto-Encoder Convolutional Non-Negative Matrix Factorization for Gestural Unit Discovery

## Datasets

First, download EMA data from http://tts.speech.cs.cmu.edu/gopalakr/emadata.tgz, untar it. Check .gitignore for the path of "emadata". 

Second, convert the normalized ema kinematics data(nema) to npy.

```
python ./src/preprocess/nema2npy.py
```

## Train

segment_len is number of ema points that is used during training. It is fixed. If sampling rate is 200Hz, it should be 200 for 1 second utterance. 
win_size is the window size in vanilla CSNMF model(in the proposal). It is also one dimension of the 2D kernel in our 2D Conv layer.
num_gestures is the number of gestures. It is typically "related" or "close to" the number of phonemes(39 in CMU phn dictionary) or acoustic units(300 in HuBERT).  

```
python ./src/main.py --segment_len 500 --win_size 10 --batch_size 16 --num_gestures 100
```
