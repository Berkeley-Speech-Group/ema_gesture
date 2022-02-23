import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = 'data/rtMRI'

all_ema = []

for spk_id in tqdm(os.listdir(path)):
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        ema_data = np.load(npy_path) #[T, 170, 2]
        if ema_data.shape[1] > 170:
            ema_data = ema_data[:,:170,:]
        ema_data = ema_data.reshape(ema_data.shape[0], -1) #[T, 340]
        all_ema.append(ema_data)

all_ema = np.concatenate(all_ema, axis=0) #[1421115, 340]
means = np.nanmean(all_ema, axis=0).reshape(1, -1) #[1, 340]
stds = np.nanstd(all_ema, axis=0).reshape(1, -1) #[1, 340]


np.save(os.path.join(path, "mean"), means)
np.save(os.path.join(path, "stds"), stds)

print("means", means)
print("stds", stds)

print("Finished Means and Stds")

for spk_id in tqdm(os.listdir(path)):
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    track_path = os.path.join(spk_id_path, 'tracks')
    for npy_file in tqdm(os.listdir(track_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(track_path, npy_file)
        ema_data = np.load(npy_path) #[T, 170, 2]
        if ema_data.shape[1] > 170:
            ema_data = ema_data[:,:170,:]
        ema_data = ema_data.reshape(ema_data.shape[0], -1) #[T, 340]
        ema_data_norm = (ema_data - means) / stds
        
        np.save(npy_path[:-4], ema_data_norm)
        
print("Norm finished")