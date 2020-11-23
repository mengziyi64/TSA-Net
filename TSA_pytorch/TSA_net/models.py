import torch.nn.functional as F
from my_tools import *
from TSA_Module import TSA_Transform
import torch
import torchvision

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    
class TSA_net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, flag_res=True):
        super(TSA_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear, flag_res)
        self.transform3 = TSA_Transform((64,64),256,256,8,(64,80),[0,0])
        self.up3 = Up(256, 128 // factor, bilinear, flag_res)
        self.transform2 = TSA_Transform((128,128),128,128,8,(64,40),[1,0])
        self.up4 = Up(128, 64, bilinear, flag_res)
        self.transform1 = TSA_Transform((256,256),64,28,8,(48,30),[1,1],True)
        self.outc = OutConv(28, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x,_ = self.transform3(x)
        x = self.up3(x, x2)
        x,_ = self.transform2(x)
        x = self.up4(x, x1)
        x,_ = self.transform1(x)
        logits = self.outc(x)
        return logits
