import torch
import torch.nn as nn
from architecture.TSA_Module import TSA_Transform
from architecture.ResidualFeat import Res2Net
from architecture.netunit import *

import pdb

_NORM_BONE = False


class TSA_Net(nn.Module):

    def __init__(self,in_ch=28, out_ch=28):
        super(TSA_Net, self).__init__()
                
        self.tconv_down1 = Encoder_Triblock(in_ch, 64, False)
        self.tconv_down2 = Encoder_Triblock(64, 128, False)
        self.tconv_down3 = Encoder_Triblock(128, 256)
        self.tconv_down4 = Encoder_Triblock(256, 512) 

        self.bottom1 = conv_block(512,1024)
        self.bottom2 = conv_block(1024,1024)

        self.tconv_up4 = Decoder_Triblock(1024, 512) 
        self.tconv_up3 = Decoder_Triblock(512, 256)
        self.transform3 = TSA_Transform((64,64),256,256,8,(64,80),[0,0])
        self.tconv_up2 = Decoder_Triblock(256, 128)
        self.transform2 = TSA_Transform((128,128),128,128,8,(64,40),[1,0])
        self.tconv_up1 = Decoder_Triblock(128, 64) 
        self.transform1 = TSA_Transform((256,256),64,28,8,(48,30),[1,1],True)
        
        self.conv_last = nn.Conv2d(out_ch, out_ch, 1)
        self.afn_last = nn.Sigmoid()
        
        
    def forward(self, x):
        enc1,enc1_pre = self.tconv_down1(x)
        enc2,enc2_pre = self.tconv_down2(enc1)
        enc3,enc3_pre = self.tconv_down3(enc2)
        enc4,enc4_pre = self.tconv_down4(enc3)
        #enc5,enc5_pre = self.tconv_down5(enc4)

        bottom = self.bottom1(enc4)
        bottom = self.bottom2(bottom)

        #dec5 = self.tconv_up5(bottom,enc5_pre)
        dec4 = self.tconv_up4(bottom,enc4_pre)
        dec3 = self.tconv_up3(dec4,enc3_pre)
        dec3,_ = self.transform3(dec3)
        dec2 = self.tconv_up2(dec3,enc2_pre)
        dec2,_ = self.transform2(dec2)
        dec1 = self.tconv_up1(dec2,enc1_pre)
        dec1,_ = self.transform1(dec1)

        dec1 = self.conv_last(dec1)
        output = self.afn_last(dec1)
        
        return output


class Encoder_Triblock(nn.Module):
    def __init__(self,inChannel,outChannel,flag_res=True,nKernal=3,nPool=2,flag_Pool=True):
        super(Encoder_Triblock, self).__init__()

        self.layer1 = conv_block(inChannel,outChannel,nKernal,flag_norm=_NORM_BONE)
        if flag_res:
            self.layer2 = Res2Net(outChannel,int(outChannel/4))
        else:
            self.layer2 = conv_block(outChannel,outChannel,nKernal,flag_norm=_NORM_BONE)

        self.pool = nn.MaxPool2d(nPool) if flag_Pool else None
    def forward(self,x):
        feat = self.layer1(x)
        feat = self.layer2(feat)

        feat_pool = self.pool(feat) if self.pool is not None else feat            
        return feat_pool,feat

class Decoder_Triblock(nn.Module):
    def __init__(self,inChannel,outChannel,flag_res=True,nKernal=3,nPool=2,flag_Pool=True):
        super(Decoder_Triblock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        if flag_res:
            self.layer2 = Res2Net(int(outChannel*2),int(outChannel/2))
        else:
            self.layer2 = conv_block(outChannel*2,outChannel*2,nKernal,flag_norm=_NORM_BONE)
        self.layer3 = conv_block(outChannel*2,outChannel,nKernal,flag_norm=_NORM_BONE)

    def forward(self,feat_dec,feat_enc):
        feat_dec = self.layer1(feat_dec)
        diffY = feat_enc.size()[2] - feat_dec.size()[2]
        diffX = feat_enc.size()[3] - feat_dec.size()[3]
        if diffY != 0 or diffX != 0:
            print('Padding for size mismatch ( Enc:', feat_enc.size(), 'Dec:', feat_dec.size(),')')
            feat_dec = F.pad(feat_dec, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        feat = torch.cat([feat_dec,feat_enc],dim=1)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        return feat