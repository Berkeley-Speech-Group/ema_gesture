import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = 'data/ieee'

all_ema = []

meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "F01" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    spk_id_path = os.path.join(spk_id_path, 'data')
    for npy_file in tqdm(os.listdir(spk_id_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(spk_id_path, npy_file)
        meta_list.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/ieee/train_metalist_F0.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/ieee/test_metalist_F0.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")

