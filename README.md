# Gestural Unit Discovery

A Pytorch Implementation of (Varitional) Auto-Encoder Convolutional Non-Negative Matrix Factorization for Gestural Unit Discovery

## Datasets

EMA

```
python ./src/preprocess/nema2npy.py
```

### Train


```
python ./src/main.py --segment_len 500 --win_size 10 --batch_size 16 --num_gestures 100
```
