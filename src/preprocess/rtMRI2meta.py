import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = 'data/rtMRI'


meta_list_all = []

#F_18 
meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "F_18" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        meta_list.append(npy_path)
        meta_list_all.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/rtMRI/train_metalist_F_18.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/rtMRI/test_metalist_F_18.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")
        
#F_25
meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "F_25" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        meta_list.append(npy_path)
        meta_list_all.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/rtMRI/train_metalist_F_25.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/rtMRI/test_metalist_F_25.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")
        
#F_28
meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "F_28" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        meta_list.append(npy_path)
        meta_list_all.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/rtMRI/train_metalist_F_28.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/rtMRI/test_metalist_F_28.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")
        

#F_29
meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "F_29" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        meta_list.append(npy_path)
        meta_list_all.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/rtMRI/train_metalist_F_29.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/rtMRI/test_metalist_F_29.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")
        
        
#M_18
meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "M_18" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        meta_list.append(npy_path)
        meta_list_all.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/rtMRI/train_metalist_M_18.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/rtMRI/test_metalist_M_18.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")
        
#M_23
meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "M_23" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        meta_list.append(npy_path)
        meta_list_all.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/rtMRI/train_metalist_M_23.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/rtMRI/test_metalist_M_23.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")
        
#M_27
meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "M_27" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        meta_list.append(npy_path)
        meta_list_all.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/rtMRI/train_metalist_M_27.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/rtMRI/test_metalist_M_27.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")
        
#M_36
meta_list = []

for spk_id in tqdm(os.listdir(path)):
    if not "M_36" in spk_id:
        continue
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        meta_list.append(npy_path)
        meta_list_all.append(npy_path)

data_size = len(meta_list)
meta_list_train = meta_list[:int(0.8*data_size)]
meta_list_test = meta_list[int(0.8*data_size):]
        
with open("data/rtMRI/train_metalist_M_36.txt", 'w') as f:
    for i in range(len(meta_list_train)):
        f.write(meta_list_train[i]+"\n")
            
with open("data/rtMRI/test_metalist_M_36.txt", 'w') as f:
    for i in range(len(meta_list_test)):
        f.write(meta_list_test[i]+"\n")
        
        
        
#all
        
data_size_all = len(meta_list_all)
meta_list_train_all = meta_list_all[:int(0.8*data_size_all)]
meta_list_test_all = meta_list_all[int(0.8*data_size_all):]
        
with open("data/rtMRI/train_metalist_all.txt", 'w') as f:
    for i in range(len(meta_list_train_all)):
        f.write(meta_list_train_all[i]+"\n")
            
with open("data/rtMRI/test_metalist_all.txt", 'w') as f:
    for i in range(len(meta_list_test_all)):
        f.write(meta_list_test_all[i]+"\n")

