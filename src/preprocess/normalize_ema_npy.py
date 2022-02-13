import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = 'data/emadata'

global_min = 1e6*np.ones(12)
global_max = -1e6*np.ones(12)

global_min = 1e6
global_max = -1e6

feature_0 = []
feature_1 = []
feature_2 = []
feature_3 = []
feature_4 = []
feature_5 = []
feature_6 = []
feature_7 = []
feature_8 = []
feature_9 = []
feature_10 = []
feature_11 = []
feature_12 = []
feature_all = []

for spk_id in tqdm(os.listdir(path)):
    if not spk_id.startswith('cin'):
        continue
    if 'mngu0' not in spk_id:
        continue
    print(spk_id)
    spk_id_path = os.path.join(path, spk_id)
    ema_dir = os.path.join(spk_id_path, "nema")

    for ema in os.listdir(ema_dir):
        if not ema.endswith('.npy'):
            continue
        ema_path = os.path.join(ema_dir, ema)
        ema_npy = np.load(ema_path) #[T, 12]
        zero_npy = np.zeros_like(ema_npy)
        #ema_npy = np.where(np.abs(ema_npy)>6, zero_npy, ema_npy)
        feature_0.append(ema_npy[0]) #[12]
        feature_1.append(ema_npy[1]) #[12]
        feature_2.append(ema_npy[2]) #[12]
        feature_3.append(ema_npy[3]) #[12]
        feature_4.append(ema_npy[4]) #[12]
        feature_5.append(ema_npy[5]) #[12]
        feature_6.append(ema_npy[6]) #[12]
        feature_7.append(ema_npy[7]) #[12]
        feature_8.append(ema_npy[8]) #[12]
        feature_9.append(ema_npy[9]) #[12]
        feature_10.append(ema_npy[10]) #[12]
        feature_11.append(ema_npy[11]) #[12]
        #global_min = np.minimum(global_min, np.min(ema_npy, axis=0))
        #global_max = np.maximum(global_max, np.max(ema_npy, axis=0))
        feature_all.append(ema_npy.reshape(-1))
        global_min = min(global_min, np.min(ema_npy))
        global_max = max(global_max, np.max(ema_npy))

print("global min is", global_min)
print('global max is', global_max)

feature_0 = np.concatenate(feature_0, axis=0)
feature_1 = np.concatenate(feature_1, axis=0)
feature_2 = np.concatenate(feature_2, axis=0)
feature_3 = np.concatenate(feature_3, axis=0)
feature_4 = np.concatenate(feature_4, axis=0)
feature_5 = np.concatenate(feature_5, axis=0)
feature_6 = np.concatenate(feature_6, axis=0)
feature_7 = np.concatenate(feature_7, axis=0)
feature_8 = np.concatenate(feature_8, axis=0)
feature_9 = np.concatenate(feature_9, axis=0)
feature_10 = np.concatenate(feature_10, axis=0)
feature_11 = np.concatenate(feature_11, axis=0)
feature_all = np.concatenate(feature_all, axis=0)

plt.hist(feature_0, 20)
plt.title("feature_0, before")
plt.savefig("feature_0_before.png")
plt.clf()

plt.hist(feature_1, 20)
plt.title("feature_1, before")
plt.savefig("feature_1_before.png")
plt.clf()

plt.hist(feature_2, 20)
plt.title("feature_2, before")
plt.savefig("feature_2_before.png")
plt.clf()

plt.hist(feature_3, 20)
plt.title("feature_3, before")
plt.savefig("feature_3_before.png")
plt.clf()

plt.hist(feature_4, 20)
plt.title("feature_4, before")
plt.savefig("feature_4_before.png")
plt.clf()

plt.hist(feature_5, 20)
plt.title("feature_5, before")
plt.savefig("feature_5_before.png")
plt.clf()

plt.hist(feature_6, 20)
plt.title("feature_6, before")
plt.savefig("feature_6_before.png")
plt.clf()

plt.hist(feature_7, 20)
plt.title("feature_7, before")
plt.savefig("feature_7_before.png")
plt.clf()

plt.hist(feature_8, 20)
plt.title("feature_8, before")
plt.savefig("feature_8_before.png")
plt.clf()

plt.hist(feature_9, 20)
plt.title("feature_9, before")
plt.savefig("feature_9_before.png")
plt.clf()

plt.hist(feature_10, 20)
plt.title("feature_10, before")
plt.savefig("feature_10_before.png")
plt.clf()

plt.hist(feature_11, 20)
plt.title("feature_11, before")
plt.savefig("feature_11_before.png")
plt.clf()

plt.hist(feature_all, 20)
plt.title("feature_all, before")
plt.savefig("feature_all_before.png")
plt.clf()

global_min_after = 1e6*np.ones(12)
global_max_after = -1e6*np.ones(12)

feature_0 = []
feature_1 = []
feature_2 = []
feature_3 = []
feature_4 = []
feature_5 = []
feature_6 = []
feature_7 = []
feature_8 = []
feature_9 = []
feature_10 = []
feature_11 = []
feature_12 = []
feature_all = []

for spk_id in tqdm(os.listdir(path)):
    if not spk_id.startswith('cin'):
        continue
    if 'mngu0' not in spk_id:
        continue
    spk_id_path = os.path.join(path, spk_id)
    ema_dir = os.path.join(spk_id_path, "nema")

    for ema in os.listdir(ema_dir):
        if not ema.endswith('.npy'):
            continue
        ema_path = os.path.join(ema_dir, ema)
        ema_npy = np.load(ema_path)
        #ema_npy = (ema_npy - global_min.reshape(1, -1)) / (global_max.reshape(1, -1) - global_min.reshape(1, -1))
        #ema_npy = (ema_npy - global_min.reshape(1, -1))
       
        ema_npy = ema_npy - global_min
        feature_0.append(ema_npy[0])
        feature_1.append(ema_npy[1])
        feature_2.append(ema_npy[2])
        feature_3.append(ema_npy[3])
        feature_4.append(ema_npy[4])
        feature_5.append(ema_npy[5])
        feature_6.append(ema_npy[6])
        feature_7.append(ema_npy[7])
        feature_8.append(ema_npy[8])
        feature_9.append(ema_npy[9])
        feature_10.append(ema_npy[10])
        feature_11.append(ema_npy[11])
        feature_all.append(ema_npy.reshape(-1))

        #global_min_after = np.minimum(global_min_after, np.min(ema_npy, axis=0))
        #global_max_after = np.maximum(global_max_after, np.max(ema_npy, axis=0))
        np.save(ema_path, ema_npy)

#print("global min after scaling is", global_min_after)
#print('global max after scaling is', global_max_after)

feature_0 = np.concatenate(feature_0, axis=0)
feature_1 = np.concatenate(feature_1, axis=0)
feature_2 = np.concatenate(feature_2, axis=0)
feature_3 = np.concatenate(feature_3, axis=0)
feature_4 = np.concatenate(feature_4, axis=0)
feature_5 = np.concatenate(feature_5, axis=0)
feature_6 = np.concatenate(feature_6, axis=0)
feature_7 = np.concatenate(feature_7, axis=0)
feature_8 = np.concatenate(feature_8, axis=0)
feature_9 = np.concatenate(feature_9, axis=0)
feature_10 = np.concatenate(feature_10, axis=0)
feature_11 = np.concatenate(feature_11, axis=0)
feature_all = np.concatenate(feature_all, axis=0)
plt.hist(feature_0, 20)
plt.title("feature_0, after")
plt.savefig("feature_0_after.png")
plt.clf()

plt.hist(feature_1, 20)
plt.title("feature_1, after")
plt.savefig("feature_1_after.png")
plt.clf()

plt.hist(feature_2, 20)
plt.title("feature_2, after")
plt.savefig("feature_2_after.png")
plt.clf()

plt.hist(feature_3, 20)
plt.title("feature_3, after")
plt.savefig("feature_3_after.png")
plt.clf()

plt.hist(feature_4, 20)
plt.title("feature_4, after")
plt.savefig("feature_4_after.png")
plt.clf()

plt.hist(feature_5, 20)
plt.title("feature_5, after")
plt.savefig("feature_5_after.png")
plt.clf()

plt.hist(feature_6, 20)
plt.title("feature_6, after")
plt.savefig("feature_6_after.png")
plt.clf()

plt.hist(feature_7, 20)
plt.title("feature_7, after")
plt.savefig("feature_7_after.png")
plt.clf()

plt.hist(feature_8, 20)
plt.title("feature_8, after")
plt.savefig("feature_8_after.png")
plt.clf()

plt.hist(feature_9, 20)
plt.title("feature_9, after")
plt.savefig("feature_9_after.png")
plt.clf()

plt.hist(feature_10, 20)
plt.title("feature_10, after")
plt.savefig("feature_10_after.png")
plt.clf()

plt.hist(feature_11, 20)
plt.title("feature_11, after")
plt.savefig("feature_11_after.png")
plt.clf()

plt.hist(feature_all, 20)
plt.title("feature_all, after")
plt.savefig("feature_all_after.png")
plt.clf()