import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from architecture.netunit import *
import numpy as np
import pdb

_NORM_ATTN = True
_NORM_FC = False

class TSA_Transform(nn.Module):
    """ Spectral-Spatial Self-Attention """
    def __init__(self, uSpace, inChannel, outChannel, nHead, uAttn, mode=[0,1], flag_mask=False, gamma_learn=False):
        super(TSA_Transform,self).__init__()
        ''' ------------------------------------------
        uSpace:
            uHeight: the [-2] dim of the 3D tensor
            uWidth: the [-1] dim of the 3D tensor
        inChannel: 
            the number of Channel of the input tensor
        outChannel: 
            the number of Channel of the output tensor
        nHead: 
            the number of Head of the input tensor
        uAttn:
            uSpatial: the dim of the spatial features
            uSpectral: the dim of the spectral features
        mask:
            The Spectral Smoothness Mask
        {mode} and {gamma_learn} is just for variable selection
        ------------------------------------------ ''' 

        self.nHead = nHead
        self.uAttn = uAttn
        self.outChannel = outChannel
        self.uSpatial = nn.Parameter(torch.tensor(float(uAttn[0])),requires_grad=False)
        self.uSpectral = nn.Parameter(torch.tensor(float(uAttn[1])),requires_grad=False)
        self.mask = nn.Parameter(Spectral_Mask(outChannel),requires_grad=False) if flag_mask else None
        self.attn_scale = nn.Parameter(torch.tensor(1.1),requires_grad=False) if flag_mask else None
        self.gamma = nn.Parameter(torch.tensor(1.0),requires_grad=gamma_learn)

        if sum(mode) > 0:
            down_sample = []
            scale = 1
            cur_channel = outChannel
            for i in range(sum(mode)):
                scale *= 2
                down_sample.append(conv_block(cur_channel,2*cur_channel,3,2,1,_NORM_ATTN))
                cur_channel = 2 * cur_channel
            self.cur_channel = cur_channel
            self.down_sample = nn.Sequential(*down_sample)
            self.up_sample = nn.ConvTranspose2d(outChannel*scale,outChannel,scale,scale)
        else:
            self.down_sample = None
            self.up_sample = None

        spec_dim = int(uSpace[0]/4-3) * int(uSpace[1]/4-3)
        self.preproc = conv1x1_block(inChannel,outChannel,_NORM_ATTN)
        self.query_x = Feature_Spatial(outChannel,nHead,int(uSpace[1]/4),uAttn[0],mode)
        self.query_y = Feature_Spatial(outChannel,nHead,int(uSpace[0]/4),uAttn[0],mode)
        self.query_lambda = Feature_Spectral(outChannel,nHead,spec_dim,uAttn[1])
        self.key_x = Feature_Spatial(outChannel,nHead,int(uSpace[1]/4),uAttn[0],mode)
        self.key_y = Feature_Spatial(outChannel,nHead,int(uSpace[0]/4),uAttn[0],mode)
        self.key_lambda = Feature_Spectral(outChannel,nHead,spec_dim,uAttn[1])
        self.value = conv1x1_block(outChannel,nHead*outChannel,_NORM_ATTN)
        self.aggregation = nn.Linear(nHead*outChannel, outChannel)
    
    def forward(self,image):
        feat = self.preproc(image)
        feat_qx = self.query_x(feat,'X')
        feat_qy = self.query_y(feat,'Y')
        feat_qlambda = self.query_lambda(feat)
        feat_kx = self.key_x(feat,'X')
        feat_ky = self.key_y(feat,'Y')
        feat_klambda = self.key_lambda(feat)
        feat_value = self.value(feat)

        feat_qx = torch.cat(torch.split(feat_qx,1,dim=1)).squeeze(dim=1)
        feat_qy = torch.cat(torch.split(feat_qy,1,dim=1)).squeeze(dim=1)
        feat_kx = torch.cat(torch.split(feat_kx,1,dim=1)).squeeze(dim=1)
        feat_ky = torch.cat(torch.split(feat_ky,1,dim=1)).squeeze(dim=1)
        feat_qlambda = torch.cat(torch.split(feat_qlambda,self.uAttn[1],dim=-1))
        feat_klambda = torch.cat(torch.split(feat_klambda,self.uAttn[1],dim=-1))
        feat_value = torch.cat(torch.split(feat_value,self.outChannel,dim=1))
        
        energy_x =  torch.bmm(feat_qx,feat_kx.permute(0,2,1))/torch.sqrt(self.uSpatial)
        energy_y =  torch.bmm(feat_qy,feat_ky.permute(0,2,1))/torch.sqrt(self.uSpatial)
        energy_lambda = torch.bmm(feat_qlambda,feat_klambda.permute(0,2,1))/torch.sqrt(self.uSpectral)

        attn_x = F.softmax(energy_x,dim=-1)
        attn_y = F.softmax(energy_y,dim=-1)
        attn_lambda = F.softmax(energy_lambda,dim=-1)
        if self.mask is not None:
            attn_lambda = (attn_lambda+self.mask)/torch.sqrt(self.attn_scale)
        
        pro_feat = feat_value if self.down_sample is None else self.down_sample(feat_value)
        batchhead,dim_c,dim_x,dim_y = pro_feat.size()
        attn_x_repeat = attn_x.unsqueeze(dim=1).repeat(1,dim_c,1,1).view(-1,dim_x,dim_x)
        attn_y_repeat = attn_y.unsqueeze(dim=1).repeat(1,dim_c,1,1).view(-1,dim_y,dim_y)
        pro_feat = pro_feat.view(-1,dim_x,dim_y)
        pro_feat = torch.bmm(pro_feat,attn_y_repeat.permute(0,2,1))
        pro_feat = torch.bmm(pro_feat.permute(0,2,1),attn_x_repeat.permute(0,2,1)).permute(0,2,1)
        pro_feat = pro_feat.view(batchhead,dim_c,dim_x,dim_y)

        if self.up_sample is not None:
            pro_feat = self.up_sample(pro_feat)
        _,_,dim_x,dim_y = pro_feat.size()
        pro_feat = pro_feat.contiguous().view(batchhead,self.outChannel,-1).permute(0,2,1)
        pro_feat = torch.bmm(pro_feat,attn_lambda.permute(0,2,1)).permute(0,2,1)
        pro_feat = pro_feat.view(batchhead,self.outChannel,dim_x,dim_y)
        pro_feat = torch.cat(torch.split(pro_feat,int(batchhead/self.nHead),dim=0),dim=1).permute(0,2,3,1)
        pro_feat = self.aggregation(pro_feat).permute(0,3,1,2)
        out = self.gamma*pro_feat + feat
        return out,(attn_x,attn_y,attn_lambda)

class Feature_Spatial(nn.Module):
    """ Spatial Feature Generation Component """
    def __init__(self, inChannel, nHead, shiftDim, outDim, mode):
        super(Feature_Spatial,self).__init__()
        kernel = [(1,5),(3,5)]
        stride = [(1,2),(2,2)]
        padding = [(0,2),(1,2)]
        self.conv1 = conv_block(inChannel,nHead,kernel[mode[0]],stride[mode[0]],padding[mode[0]],_NORM_ATTN)
        self.conv2 = conv_block(nHead,nHead,kernel[mode[1]],stride[mode[1]],padding[mode[1]],_NORM_ATTN)
        self.fully = fully_block(shiftDim,outDim,_NORM_FC)
    def forward(self,image,direction):
        if direction == 'Y':
            image = image.permute(0,1,3,2)
        feat = self.conv1(image)
        feat = self.conv2(feat)
        feat = self.fully(feat)
        return feat

class Feature_Spectral(nn.Module):
    """ Spectral Feature Generation Component """
    def __init__(self, inChannel, nHead, viewDim, outDim):
        super(Feature_Spectral,self).__init__()
        self.inChannel = inChannel
        self.conv1 = conv_block(inChannel,inChannel,5,2,0,_NORM_ATTN)
        self.conv2 = conv_block(inChannel,inChannel,5,2,0,_NORM_ATTN)
        self.fully = fully_block(viewDim,int(nHead*outDim),_NORM_FC)
    def forward(self,image):
        bs = image.size(0)
        feat = self.conv1(image)
        feat = self.conv2(feat)
        feat = feat.view(bs,self.inChannel,-1)
        feat = self.fully(feat)
        return feat

def Spectral_Mask(dim_lambda):
    '''After put the available data into the model, we use this mask to avoid outputting the estimation of itself.'''
    orig = (np.cos(np.linspace(-1, 1, num=2*dim_lambda-1)*np.pi)+1.0)/2.0
    att = np.zeros((dim_lambda,dim_lambda))
    for i in range(dim_lambda):
        att[i,:] = orig[dim_lambda-1-i:2*dim_lambda-1-i]
    AM_Mask = torch.from_numpy(att.astype(np.float32)).unsqueeze(0)
    return AM_Mask

