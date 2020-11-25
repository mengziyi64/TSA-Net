import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from spectral import SpectralNorm
import numpy as np

class Res2Net(nn.Module):
    def __init__(self, inChannel, uPlane, scale=4):
        super(Res2Net, self).__init__()
        self.uPlane = uPlane
        self.scale = scale

        self.conv_init = nn.Conv2d(inChannel, uPlane*scale, kernel_size=1, bias=False)
        self.bn_init = nn.BatchNorm2d(uPlane*scale)

        convs = []
        bns = []
        for i in range(self.scale-1):
            convs.append(nn.Conv2d(self.uPlane, self.uPlane, kernel_size=3, stride = 1, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(self.uPlane))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv_end = nn.Conv2d(uPlane*scale, inChannel, kernel_size=1, bias=False)
        self.bn_end = nn.BatchNorm2d(inChannel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):

        out = self.conv_init(x)
        out = self.bn_init(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.uPlane, 1)
        for i in range(self.scale-1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.scale-1]),1)

        out = self.conv_end(out)
        out = self.bn_end(out)
        return out


'''
------------------------------------------- Original Res2Net Version --------------------------------------------- 
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''