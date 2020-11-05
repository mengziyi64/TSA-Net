from torch.utils import data
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from utils import *
class dataset(data.Dataset):
    
    
    def __init__(self, file_path, mask3d, total_batch_num):
        
        self.data_list = LoadTraining(file_path)
        self.mask = mask3d  
        self.total_batch_num = total_batch_num
        
    def __len__(self):
        return self.total_batch_num
    
    def __getitem__(self, index):
    
        idx = np.random.randint(len(self.data_list), size = 1)[0]
        processed_data = np.zeros((1, 256, 256, 28), dtype=np.float32)
        h, w, _ = self.data_list[idx].shape
        x_index = np.random.randint(h - 256, size = 1)[0]
        y_index = np.random.randint(w - 256, size = 1)[0]
        processed_data[0] = self.data_list[idx][x_index:x_index + 256, y_index:y_index + 256, :]
        
        train_truth = processed_data
        #train_truth[0] = train_truth[0]/train_truth[0].max()
        rot_angle = random.randint(1,4)
        train_truth[0] = np.rot90(train_truth[0], rot_angle)

        train_PhiTy = np.transpose(geneMeasurement(train_truth, self.mask), (0, 3, 1, 2))
        train_truth = np.transpose(train_truth, (0, 3, 1, 2))
    
        return train_truth[0], train_PhiTy[0]
    
