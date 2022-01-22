# Gestural Unit Discovery

A Pytorch Implementation of (Varitional) Auto-Encoder Convolutional Non-Negative Matrix Factorization for Gestural Unit Discovery

## Tracking

### Slides

https://docs.google.com/presentation/d/10XbOxXiPCrw9Go2Qp_-aBY7Cp7l6JDoCT87a2W8VhpM/edit?usp=sharing

### Docs

https://www.dropbox.com/scl/fi/orhv9g851sfftdbr0lssz/Gestural-Unit-Discovery2022-NAACL-Jan16Jan15-JSTSP.paper?dl=0&rlkey=bbhoawttjstpbl255mutxk8df

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

## Datasets

First, download EMA data from http://tts.speech.cs.cmu.edu/gopalakr/emadata.tgz, extract it. Check .gitignore for the path of "emadata". Note that we use "nema" data, which is the standardized version of the original data. There are 4579 wav utterances but 4409 ema kinematics data. To handle such mismatch, we only take the ema data that has its corresponding waveform. Actually the first stage(Task 1 of proposal) focuses on gestural unit discovery and only ema data is used.  

Second, convert the standardized ema kinematics data(nema) to npy.

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

##  Phoneme Recognition on EMA

```
python src/main.py --pr_ema --save_path save_models/pr_ema_bs8
```

##  Phoneme Recognition on MFCC

```
python src/main.py --pr_mfcc --save_path save_models/pr_mfcc_bs8
```

##  Phoneme Recognition on STFT

```
python src/main.py --pr_stft --save_path save_models/pr_stft_bs8
```


## Launch Tensorboard

```
tensorboard --logdir=runs
```


## Test | Visuallize Gestures, Reconstructions and Mel_Spec


```
python src/main.py --vis_gestures --model_path save_models/test/best.pth --resynthesis --test_ema_path emadata/cin_us_mngu0/nema/mngu0_s1_0300.npy --spk_id mngu0
```
