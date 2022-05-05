# Gestural Unit Discovery

A Pytorch Implementation of (Varitional) Auto-Encoder Convolutional Non-Negative Matrix Factorization for Gestural Unit Discovery




## Environents

```
conda create --name ema
conda activate ema
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Install Packages:

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
cd ..
conda install -c conda-forge librosa
conda install -c conda-forge tensorboard
pip install Levenshtein
pip install seaborn

```

## Data


```
python ./src/preprocess/nema_label2npy.py
```

Third, re-normalize the data within [0, 1] because we are performing NMF. (Right now this step is not required)

```
python ./src/preprocess/normalize_ema_npy.py
```

Fourth, run kmeans on ema huge and supervector:

```
python ./src/kmeans.py
```


## Train Resynthesis

```
python src/main.py --sparse_c --sparse_c_factor 10 --spk_id mngu --sparse_t --sparse_t_factor 100 --learning_rate 1e-3 --batch_size 8 --sparse_c_base 0.90 --segment_len 350 --entropy_t --entropy_t_factor 10 --entropy_c --entropy_c_factor 1 --pr_joint --resynthesis
```

##  Resynthesis + CTC on H joint training

```
python src/main.py --sparse_c --sparse_c_factor 10 --spk_id mngu --sparse_t --sparse_t_factor 100 --learning_rate 1e-3 --batch_size 8 --sparse_c_base 0.90 --segment_len 350 --entropy_t --entropy_t_factor 10 --entropy_c --entropy_c_factor 1 --pr_joint --resynthesis
```

##  Phoneme Recognition on Melspectrogram

```
python src/main.py --pr_mel --save_path save_models/pr_mel_bs8
```
