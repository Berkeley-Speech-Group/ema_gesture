import os
import numpy as np

path = 'emadata'

global_min = 1e6
global_max = -1e6
for spk_id in os.listdir(path):
    if not spk_id.startswith('cin'):
        continue
    spk_id_path = os.path.join(path, spk_id)
    ema_dir = os.path.join(spk_id_path, "nema")

    for ema in os.listdir(ema_dir):
        if not ema.endswith('.npy'):
            continue
        ema_path = os.path.join(ema_dir, ema)
        ema_npy = np.load(ema_path)
        global_min = min(global_min, np.min(ema_npy))
        global_max = max(global_max, np.max(ema_npy))

for spk_id in os.listdir(path):
    if not spk_id.startswith('cin'):
        continue
    spk_id_path = os.path.join(path, spk_id)
    ema_dir = os.path.join(spk_id_path, "nema")

    for ema in os.listdir(ema_dir):
        if not ema.endswith('.npy'):
            continue
        ema_path = os.path.join(ema_dir, ema)
        ema_npy = np.load(ema_path)
        ema_npy = ema_npy - global_min
        np.save(ema_path, ema_npy)
                    