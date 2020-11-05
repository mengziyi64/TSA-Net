import scipy.io as sio
import os 
import numpy as np
import matplotlib.pyplot as plt
import math
import torch 
import logging
from ssim_torch import ssim


def generate_masks(mask_path):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,28))
    #mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = mask3d[np.newaxis,:,:,:]
    #mask3d = torch.from_numpy(mask3d)
    #mask3d = mask3d.float()
    #mask3d = mask3d.cuda()

    return mask3d

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    print('training sences:', len(scene_list))
    max_ = 0
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs

def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)['img']
        #img = img/img.max()
        test_data[i,:,:,:] = img
        print(i, img.shape, img.max(), img.min())
    return test_data

def psnr(img1, img2):
    psnr_list = []
    for i in range(img1.shape[0]):
        total_psnr = 0
        #PIXEL_MAX = img2.max()
        PIXEL_MAX = img2[i,:,:,:].max()
        for ch in range(28):
            mse = np.mean((img1[i,:,:,ch] - img2[i,:,:,ch])**2)
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        psnr_list.append(total_psnr/img1.shape[3])
    return psnr_list

def torch_psnr(img, ref):      #input [28,256,256]
    nC = img.shape[0]
    pixel_max = torch.max(ref)
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i,:,:] - ref[i,:,:]) ** 2)
        psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr/nC

def torch_ssim(img, ref):   #input [28,256,256]
    return ssim(torch.unsqueeze(img,0), torch.unsqueeze(ref,0))


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename




def shuffle_crop(train_data,batch_size):
    
    index = np.random.choice(range(len(train_data)), batch_size)
    processed_data = np.zeros((batch_size, 256, 256, 28), dtype=np.float32)
    
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape
        x_index = np.random.randint(0, h - 256)
        y_index = np.random.randint(0, w - 256)
        processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + 256, y_index:y_index + 256, :]
        

    return processed_data


def geneMeasurement(truth, mask):
    batch_size = truth.shape[0]
    temp = truth * mask
    H, W = 256, 256
    shift = 2
    temp_shift = np.zeros((batch_size, H, W+(28-1)*shift, 28), dtype=np.float32);
    temp_shift[:, :, 0:W, :] = temp
    
    for k in range(28):
        temp_shift[:, :, :, k] = np.roll(temp_shift[:, :, :, k], shift*k, axis=2)
    meas = np.sum(temp_shift, axis=3)                  #[bs, 256,310]
    meas = meas/14
    # PhiTy
    meas_temp = np.zeros((batch_size, H, W, 28), dtype=np.float32)
    for i in range(28):
        meas_temp[:, :, :, i] = meas[:, :, i*2:i*2+W]
    PhiTy = meas_temp * mask                      #[256,256,28]
    return PhiTy




def XYZ2sRGB_exgamma(XYZ):
    bs, W,H,nC = XYZ.shape
    XYZ = torch.reshape(XYZ, (bs*W*H, nC))
    M = torch.tensor([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0414], [0.0557, -0.2040, 1.0570]]).cuda(0)
    sRGB = torch.transpose(torch.matmul(M, torch.transpose(XYZ,0,1)), 0, 1)
    sRGB = torch.reshape(sRGB, (bs, W, H, nC))
    return sRGB

def cie_match(img):
    img = img.permute(0, 2, 3, 1)
    bs, W,H,nC = img.shape
    illumination = 1
    img = img*illumination
    
    img_flat = torch.reshape(img, (bs, W*H, nC))
    #CIE_use = sio.loadmat('CIE_28.mat')['CIE_28']
    CIE_use = torch.tensor([[2.69773310e-01, 5.01781189e-02, 1.40777994e+00],
                        [2.44874413e-01, 5.57073087e-02, 1.32639298e+00],
                        [2.23665444e-01, 6.47629082e-02, 1.27013206e+00],
                        [1.98846405e-01, 7.89252005e-02, 1.20056329e+00],
                        [1.61615584e-01, 9.75573676e-02, 1.06240141e+00],
                        [1.20249078e-01, 1.20229110e-01, 8.86307473e-01],
                        [7.98656180e-02, 1.48314649e-01, 6.98942215e-01],
                        [4.62666876e-02, 1.83256380e-01, 5.29410321e-01],
                        [2.25778095e-02, 2.30361677e-01, 3.91935035e-01],
                        [7.90840513e-03, 2.95033104e-01, 2.92225680e-01],
                        [2.17490999e-03, 3.85797936e-01, 2.17899615e-01],
                        [9.03314558e-03, 5.00944413e-01, 1.55435112e-01],
                        [3.62700355e-02, 6.33408947e-01, 1.00068318e-01],
                        [8.78099676e-02, 7.58386468e-01, 6.44966803e-02],
                        [1.61050842e-01, 8.55976634e-01, 4.27107972e-02],
                        [2.46713459e-01, 9.28205108e-01, 2.62761216e-02],
                        [3.45144944e-01, 9.74871784e-01, 1.47078596e-02],
                        [4.57957969e-01, 9.97366936e-01, 7.82448292e-03],
                        [5.74185600e-01, 9.97533460e-01, 4.47124362e-03],
                        [7.22959575e-01, 9.66628366e-01, 2.54760762e-03],
                        [8.48534631e-01, 9.13098766e-01, 1.95927249e-03],
                        [9.69374692e-01, 8.24155835e-01, 1.58998507e-03],
                        [1.04914196e+00, 7.02397158e-01, 1.13294834e-03],
                        [1.04088431e+00, 5.76840200e-01, 7.32833397e-04],
                        [9.37343613e-01, 4.49000957e-01, 3.25280179e-04],
                        [7.37670392e-01, 3.19345060e-01, 1.55047346e-04],
                        [5.08796071e-01, 2.05585860e-01, 6.37296306e-05],
                        [3.06017843e-01, 1.18090556e-01, 2.81370917e-05]])
    CIE_use = CIE_use.cuda(0)
    XYZ = torch.matmul(img_flat, CIE_use)

    XYZ = torch.reshape(XYZ, (bs, W, H, 3))
    XYZ = XYZ/XYZ.max()

    RGB = XYZ2sRGB_exgamma(XYZ)
    RGB = torch.clamp(RGB, min=0, max=1)
    return RGB.permute(0, 3, 1, 2)

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger