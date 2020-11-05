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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "../Data/training/"  
mask_path = "../Data"  
test_path = "../Data/testing/" 

batch_size = 10
last_train = 0                        # for finetune
model_save_filename = ''                 # for finetune
max_iter = 300
learning_rate = 0.0004
epoch_sam_num = 5000
batch_num = int(np.floor(epoch_sam_num/batch_size))
total_sample_num = epoch_sam_num*max_iter

mask3d = generate_masks(mask_path)

train_set = dataset(data_path, mask3d, total_sample_num)
data_iter = DataLoader(train_set, batch_size = batch_size, shuffle=False, num_workers=2)
test_data = LoadTest(test_path)
test_PhiTy = geneMeasurement(test_data, mask3d)
test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
test_PhiTy =torch.from_numpy(np.transpose(test_PhiTy, (0, 3, 1, 2)))

model = TSA_net(28, 28).cuda()

if last_train != 0:
    model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))
    
optimizer_G = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))
mse = torch.nn.MSELoss().cuda()
#####  train & test
best_test_psnr = 0.
count =0
if model_save_filename == '':
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
else:
    date_time = model_save_filename
result_path = 'result' + '/' + date_time
model_path = 'model' + '/' + date_time
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
logger = gen_log(model_path)
logger.info("Learning rate:{}, batch_size:{}.\n".format(learning_rate, batch_size))

def checkpoint(epoch, model_path, logger):
    model_out_path = './' + model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

begin = time.time()
epoch_loss = 0
for train_truth, train_PhiTy in data_iter:
    
    real_A = train_PhiTy.cuda()
    real_B = train_truth.cuda()
    fake_B = model(real_A)
    optimizer_G.zero_grad()
    loss = mse(fake_B, real_B).cuda()
    epoch_loss += loss.data
    
    loss.backward()
    optimizer_G.step()
    
    if count % batch_num == 0 and count != 0:
        epoch = last_train + int(np.ceil(count/batch_num))
        end = time.time()
        logger.info("===> Epoch {} Complete: Avg. Loss: {:.5f} time: {:.2f}".format(epoch, epoch_loss/batch_num, end - begin))
        epoch_loss = 0
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
        logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}'.format(epoch, psnr_mean, ssim_mean))
        if psnr_mean > best_test_psnr and psnr_mean > 27:
            best_test_psnr = psnr_mean
            sio.savemat(result_path+'/test_epoch_{}_{:.2f}_{:.3f}.mat'.format(epoch, psnr_mean, ssim_mean), {'truth':truth, 'pred':pred, 'psnr_list':psnr_list, 'ssim_list':ssim_list})
            checkpoint(epoch, model_path, logger)
        model.train()
        begin = time.time()
    count +=1