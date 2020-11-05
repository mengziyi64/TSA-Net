from dataloader import dataset
from models import TSA_net
from utils import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

mask_path = "../Data"  
test_path = "../Data/testing/" 

last_train = 0                        # the epoch num of the pretrained model
model_save_filename = ''                 # the filename for the pretrained model you saved

mask3d = generate_masks(mask_path)

test_data = LoadTest(test_path)
test_PhiTy = geneMeasurement(test_data, mask3d)
test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
test_PhiTy =torch.from_numpy(np.transpose(test_PhiTy, (0, 3, 1, 2)))

model = TSA_net(28, 28).cuda()

if last_train != 0:
    model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))
    
#####  test
if model_save_filename == '':
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
else:
    date_time = model_save_filename
result_path = 'result' + '/' + date_time
if not os.path.exists(result_path):
    os.makedirs(result_path)

logger = gen_log(model_path)
logger.info("Learning rate:{}, batch_size:{}.\n".format(learning_rate, batch_size))



### test
psnr_list, ssim_list = [], []
model.eval()
test_truth = test_data.cuda().float()
model_out = model(test_PhiTy.cuda().float())
        
for k in range(test_truth.shape[0]):
    psnr_val = torch_psnr(model_out[k,:,:,:], test_truth[k,:,:,:])
    ssim_val = torch_ssim(model_out[k,:,:,:], test_truth[k,:,:,:])
    psnr_list.append(psnr_val.detach().cpu().numpy())
    ssim_list.append(ssim_val.detach().cpu().numpy())

pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
truth = np.transpose(test_truth.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
psnr_mean = np.mean(np.asarray(psnr_list))
ssim_mean = np.mean(np.asarray(ssim_list))
logger.info(' Testing psnr = {:.2f}, ssim = {:.3f}'.format(psnr_mean, ssim_mean))

sio.savemat(result_path+'/test_epoch_{}_{:.2f}_{:.3f}.mat'.format(last_train, psnr_mean, ssim_mean), {'truth':truth, 'pred':pred, 'psnr_list':psnr_list, 'ssim_list':ssim_list})
