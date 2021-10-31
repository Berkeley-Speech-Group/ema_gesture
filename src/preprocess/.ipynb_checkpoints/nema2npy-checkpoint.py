import os
import numpy as np

path = 'emadata'
for spk_id in os.listdir(path):
    if not spk_id.startswith('cin'):
        continue
    spk_id_path = os.path.join(path, spk_id)
    ema_dir = os.path.join(spk_id_path, "nema")

    for ema in os.listdir(ema_dir):
        if not ema.endswith('.ema'):
            continue
        if ema.endswith('.ema'):
            ema_path = os.path.join(ema_dir, ema)
            #here we have ema_path
            
            ema_data = []
            with open(ema_path) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line_data = [float(line.strip().split()[i]) for i in range(12)]
                    ema_data.append(line_data)
            ema_data = np.array(ema_data)
            ema_npy_path = ema_path[:-3] + 'npy'
            np.save(ema_npy_path, ema_data)
                    