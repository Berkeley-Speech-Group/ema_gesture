import os
import numpy as np
import scipy.io
from tqdm import tqdm

path = 'data/ieee'
for spk_id in tqdm(os.listdir(path)):
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    spk_id_path = os.path.join(path, spk_id)
    spk_id_path = os.path.join(spk_id_path, 'data')
    for mat_file in os.listdir(spk_id_path):
        if not mat_file.endswith(".mat"):
            continue
        if "palate" in mat_file:
            continue
        mat_id = mat_file[:-4]
        mat_path = os.path.join(spk_id_path, mat_file)
        print(mat_path)
        
        mat = scipy.io.loadmat(mat_path)

        data = mat[mat_id]
        data = data[0]
        
        if len(data) < 9:
            continue
        
        audio = data[0]
        TR = data[1]
        TB = data[2]
        TT = data[3]
        UL = data[4]
        LL = data[5]
        ML = data[6]
        JAW = data[7]
        JAWL = data[8]
        wavform = np.array(audio[2]).reshape(-1) #{T_wav}

        TR_data = np.array(TR[2]) #[T,6]
        TB_data = np.array(TB[2]) #[T,6]
        TT_data = np.array(TT[2]) #[T,6]
        UL_data = np.array(UL[2]) #[T,6]
        LL_data = np.array(LL[2]) #[T,6]
        ML_data = np.array(ML[2]) #[T,6]
        JAW_data = np.array(JAW[2]) #[T,6]
        JAWL_data = np.array(JAWL[2]) #[T,6]

        TR_data_p = np.array(TR[2][:,:3]) #[T,3]
        TB_data_p = np.array(TB[2][:,:3]) #[T,3]
        TT_data_p = np.array(TT[2][:,:3]) #[T,3]
        UL_data_p = np.array(UL[2][:,:3]) #[T,3]
        LL_data_p = np.array(LL[2][:,:3]) #[T,3]
        ML_data_p = np.array(ML[2][:,:3]) #[T,3]
        JAW_data_p = np.array(JAW[2][:,:3]) #[T,3]
        JAWL_data_p = np.array(JAWL[2][:,:3]) #[T,3]

        ema_data_p = np.concatenate((TR_data_p, TB_data_p, TT_data_p, UL_data_p, LL_data_p, ML_data_p, JAW_data_p, JAWL_data_p), axis=-1) #[T, 24]
        
