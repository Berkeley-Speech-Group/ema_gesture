import os
import numpy as np
import scipy.io
from tqdm import tqdm

path = 'data/rtMRI'


print("Convert track mat files to npy....")

for spk_id in os.listdir(path):
    
    if not spk_id.startswith("F") and not spk_id.startswith("M"):
        continue
    print("Speaker ID is " + spk_id)
    spk_id_path = os.path.join(path, spk_id)
    tracks_dir_path = os.path.join(spk_id_path, 'tracks')
    
    for track_mat in tqdm(os.listdir(tracks_dir_path)):
        if not track_mat.endswith("mat"):
            continue
        
        track_mat_path = os.path.join(tracks_dir_path, track_mat)
        
        mat = scipy.io.loadmat(track_mat_path)
        res = mat['trackdata']
        num_frames = len(res[0])

        track_array = []

        for t in range(num_frames):
            frame_array = []
            
            frame = res[0][t]
            
            while(len(frame) == 1):
                frame = frame[0]
                
            if len(frame) == 0:
                print("Frame Missing!!!!!")
                print(track_mat_path + " frames: " + str(t))
                continue
                
            frame = frame[0]
            
            while(len(frame) == 1):
                frame = frame[0]
            
            #length is 4 right now, we only need frame[0], frame[1], frame[2]
            

            segment0 = frame[0]
            while(len(segment0) == 1):
                segment0 = segment0[0]
            segment0_v = segment0[0]
            segment0_i = segment0[1]
            segment0_mu = segment0[2]

            segment1 = frame[1]
            while(len(segment1) == 1):
                segment1 = segment1[0]
            segment1_v = segment1[0]
            segment1_i = segment1[1]
            segment1_mu = segment1[2]

            segment2 = frame[2]
            while(len(segment2) == 1):
                segment2 = segment2[0]
            segment2_v = segment2[0]
            segment2_i = segment2[1]
            segment2_mu = segment2[2]

            frame_array.append(segment0_v)
            frame_array.append(segment1_v)
            frame_array.append(segment2_v)
            
            frame_array = np.concatenate(frame_array, axis=0)
            frame_array = np.expand_dims(frame_array, axis=0)
            track_array.append(frame_array)
            
            
        track_array = np.concatenate(track_array, axis=0) #[T, 170, 2]
        #print(track_array.shape)
        np.save(track_mat_path[:-4], track_array)
        
print("Finished....")
       
            
            
            

