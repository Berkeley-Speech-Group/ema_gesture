import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = 'data/ieee'

all_ema = []

for spk_id in tqdm(os.listdir(path)):
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    spk_id_path = os.path.join(spk_id_path, 'data')
    for npy_file in tqdm(os.listdir(spk_id_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(spk_id_path, npy_file)
        ema_data = np.load(npy_path) #[T, 24]
        #print(np.max(ema_data))
        all_ema.append(ema_data)

all_ema = np.concatenate(all_ema, axis=0) #[1421115, 24]
means = np.nanmean(all_ema, axis=0).reshape(1, -1) #[1, 24]
stds = np.nanstd(all_ema, axis=0).reshape(1, -1) #[1, 24]


np.save(os.path.join(path, "mean"), means)
np.save(os.path.join(path, "stds"), stds)

print("means", means)
print("stds", stds)


print("Finished Means and Stds")

for spk_id in tqdm(os.listdir(path)):
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    spk_id_path = os.path.join(spk_id_path, 'data')
    for npy_file in tqdm(os.listdir(spk_id_path)):
        if not npy_file.endswith(".npy"):
            continue
        npy_id = npy_file[:-4]
        npy_path = os.path.join(spk_id_path, npy_file)
        ema_data = np.load(npy_path) #[T, 24]
        
        ema_data_norm = (ema_data - means) / stds
        
        np.save(npy_path[:-4], ema_data_norm)
        
print("Norm finished")