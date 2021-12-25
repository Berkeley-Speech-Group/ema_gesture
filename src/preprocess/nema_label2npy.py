import os
import numpy as np

path = 'emadata'
phn_set = set()

for spk_id in os.listdir(path):
    if not spk_id.startswith('cin'):
        continue
    spk_id_path = os.path.join(path, spk_id)
    ema_dir = os.path.join(spk_id_path, "nema")
    lab_dir = os.path.join(spk_id_path, "lab")

    #ema2npy
    for ema in os.listdir(ema_dir):
        if not ema.endswith('.ema'):
            continue

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


    #lab2npy
    for lab in os.listdir(lab_dir):
        if not lab.endswith('.lab'):
            continue

        lab_path = os.path.join(lab_dir, lab)
        #here we have ema_path
        
        lab_data = []
        with open(lab_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("#"):
                    continue
                line_data = line.strip().split()
                cur_lab = line_data[-1]
                phn_set.add(cur_lab)
                lab_data.append(cur_lab)
        lab_data = np.array(lab_data)
        lab_npy_path = lab_path[:-3] + 'npy'
        #np.save(lab_npy_path, lab_data)

print("# of phns is ", len(phn_set))
print("phn set is ", phn_set)

phn_map = {}
cnt = 0
for key in sorted(phn_set):
    phn_map[key] = cnt
    cnt += 1
PHONEME_MAP = phn_map

print("phn_map is ", phn_map)

print("after iterate over all data to obtain the phoneme map and we come back again to get numerical labels")

for spk_id in os.listdir(path):
    if not spk_id.startswith('cin'):
        continue
    spk_id_path = os.path.join(path, spk_id)
    lab_dir = os.path.join(spk_id_path, "lab")

    #lab2npy
    for lab in os.listdir(lab_dir):
        if not lab.endswith('.lab'):
            continue
        lab_path = os.path.join(lab_dir, lab)
        #here we have ema_path
        
        lab_data = []
        with open(lab_path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("#"):
                    continue
                line_data = line.strip().split()
                cur_lab = line_data[-1]
                cur_phn_idx = phn_map[cur_lab]
                lab_data.append(cur_phn_idx)
        lab_data = np.array(lab_data)
        lab_npy_path = lab_path[:-3] + 'npy'
        np.save(lab_npy_path, lab_data)

        
                    