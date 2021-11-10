# Gestural Unit Discovery

A Pytorch Implementation of (Varitional) Auto-Encoder Convolutional Non-Negative Matrix Factorization for Gestural Unit Discovery

## Datasets

First, download EMA data from http://tts.speech.cs.cmu.edu/gopalakr/emadata.tgz, untar it. Check .gitignore for the path of "emadata". 

Second, convert the normalized ema kinematics data(nema) to npy.

```
python ./src/preprocess/nema2npy.py
```

## Train


```
python ./src/main.py --segment_len 500 --win_size 10 --batch_size 16 --num_gestures 100
```
