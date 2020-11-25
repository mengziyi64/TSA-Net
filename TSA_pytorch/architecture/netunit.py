import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb

def conv_block(in_planes, out_planes, the_kernel=3, the_stride=1, the_padding=1, flag_norm=False, flag_norm_act=True):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=the_kernel, stride=the_stride, padding=the_padding)
    activation = nn.ReLU(inplace=True)
    norm = nn.BatchNorm2d(out_planes)
    if flag_norm:
        return nn.Sequential(conv,norm,activation) if flag_norm_act else nn.Sequential(conv,activation,norm)
    else:
        return nn.Sequential(conv,activation)

def conv1x1_block(in_planes, out_planes, flag_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,bias=False)
    norm = nn.BatchNorm2d(out_planes)
    return nn.Sequential(conv,norm) if flag_norm else conv

def fully_block(in_dim, out_dim, flag_norm=False, flag_norm_act=True):
    fc = nn.Linear(in_dim, out_dim)
    activation = nn.ReLU(inplace=True)
    norm = nn.BatchNorm2d(out_dim)
    if flag_norm:
        return nn.Sequential(fc,norm,activation) if flag_norm_act else nn.Sequential(fc,activation,norm)
    else:
        return nn.Sequential(fc,activation)
