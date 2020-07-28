from __future__ import division
import numpy as np
import os
import random
import scipy.io as sio
import hdf5storage
import random
import matplotlib.pyplot as plt
from PIL import Image 

def Data_Division(dataset_dir):

    nC = 28
    (data_dir, mask_dir) = dataset_dir
    if not os.path.exists(data_dir[0]):
        os.makedirs(data_dir[0])
    if not os.path.exists(data_dir[1]):
        os.makedirs(data_dir[1])
    training_list = os.listdir(data_dir[0])
    valid_list = os.listdir(data_dir[1])
    testing_list = os.listdir(data_dir[2])
    testing_list.sort()
    mask_file = sio.loadmat(mask_dir + 'mask.mat')
    mask = mask_file['mask']
    mask_3d = np.tile(mask[:,:,np.newaxis],(1,1,nC))
    return data_dir, training_list, valid_list, testing_list, mask_3d


def Data_Generator_File(data_dir, data_list, mask, batch_size, is_training=True, is_valid=False, is_testing=False):
    
    W, H ,nC= 660, 660, 28
    shift = 2
    sample_num = len(data_list)
    index = np.random.choice(sample_num, size=sample_num, replace=False).astype(np.int16)
    sample_cnt,batch_cnt,list_measure,list_ground = 0,0,[],[]
    while True:
        if (sample_cnt < sample_num):
            if is_training is True:
                ind_set = index[sample_cnt]
            else:
                ind_set = sample_cnt
            if is_testing is True:
                img = np.zeros((H,W,nC))
                meas = sio.loadmat(data_dir[2] + data_list[ind_set])['meas_real']
                meas = meas/meas.max()*0.8
            else:
                if is_training is True:  
                    img = sio.loadmat(data_dir[0] + data_list[ind_set])['img']
                    rot_angle = random.randint(1,4)
                    img = np.rot90(img, rot_angle)
                elif is_valid is True:    
                    img = sio.loadmat(data_dir[1] + data_list[ind_set])['img']
                img[img< 0] = 0
                img = img.astype(np.float32)
                img=img/65535
                temp = mask*img
                temp_shift = np.zeros((H,W+(nC-1)*shift,nC));
                temp_shift[:,0:W,:] = temp;
                for t in range(nC):
                    temp_shift[:,:,t] = np.roll(temp_shift[:,:,t],shift*t,axis=1)
                meas = np.sum(temp_shift,axis=2)
                meas = meas/nC*2*1.2
                # shot noise
                QE, bit = 0.4, 2048
                meas = np.random.binomial((meas*bit/QE).astype(int),QE)
                meas = meas/bit
            # transpose(Phi)*y
            meas_temp = np.zeros((H,W,nC))
            for i in range(nC):
                meas_temp[:,:,i] = meas[:,i*shift:i*shift+W]
            meas_temp = meas_temp*mask
            list_measure.append(meas_temp)
            list_ground.append(img)
            batch_cnt += 1
            sample_cnt += 1
                                      
            if batch_cnt == batch_size:
                batch_measure,batch_ground = np.stack(list_measure,0),np.stack(list_ground,0)
                height_init,batch_cnt,list_measure,list_ground = 0,0,[],[]
                yield batch_measure,mask,batch_ground
        else:            
            sample_cnt = 0
            index = np.random.choice(sample_num, size=sample_num, replace=False).astype(np.int16)

                
                
