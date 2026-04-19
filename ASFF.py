

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda import amp
from einops import rearrange

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os

    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box


class Conv(nn.Module): #定义卷积
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p




class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

def conv_1x1_bn(inp,oup):
    return nn.Sequential(
        nn.Conv2d(inp,oup,1,1,0,bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
class Proto(nn.Module): #这个Proto类是YOLOv5中用于分割模型的掩码Proto模块。它包括了输入通道、Proto通道和掩码通道的配置。在前向传播过程中，它使用了卷积层和上采样操作。
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):
        """Initializes YOLOv5 Proto module for segmentation with input, proto, and mask channels configuration."""
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass using convolutional layers and upsampling on input tensor `x`."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Upsample(nn. Module):



    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()

#self.cv1=Conv(c1,c2,1)

#self. upsample nn. Upsample(scale_factor=scale_factor, mode='nearest'
        if scale_factor == 2:

            self.cv1=nn.ConvTranspose2d(c1,c2,2,2,0,bias=True)#如果下采
        elif scale_factor == 4:

            self.cv1=nn.ConvTranspose2d(c1,c2,4,4,0,bias=True)#如果下采
    def forward(self, x):

#return self. upsample(self. cv1(x))
        return self.cv1(x)


#———2.自适应空间融合（ASFF）———#class ASFF2(nn. Module):
class ASFF2(nn.Module):

    def __init__(self, c1, c2, level=0):

        super().__init__()

        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim=c1_l,c1_h
        self.inter_dim=self.dim[self.level]
        compress_c=8

        if level == 0:
            self.stride_level_1=Upsample(c1_h,self.inter_dim)
        if level == 1:
            self.stride_level_0=Conv(c1_l,self.inter_dim,2,2,0)

        self.weight_level_0=Conv(self.inter_dim,compress_c, 1, 1)
        self.weight_level_1=Conv(self.inter_dim,compress_c,1,1)

        self.weights_levels=nn.Conv2d(compress_c * 2, 2, 1,1,0)
        self.conv=Conv(self.inter_dim,self.inter_dim, 3 , 1)

    def forward(self,x):
        x_level_0,x_level_1=x[0],x[1]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)

        levels_weight_v =torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weights_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight,dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1] +level_1_resized * levels_weight[:,1:2]
        return self.conv(fused_out_reduced)

class ASFF3(nn.Module):
    def __init__(self,c1,c2,level=0):
        super().__init__()
        c1_l,c1_m,c1_h = c1[0],c1[1],c1[2]
        self.level = level
        self.dim = c1_l,c1_m,c1_h
        self.inter_dim = self.dim[self.level]
        compress_c =8

        if level == 0:
            self.stride_level_1=Upsample(c1_m,self.inter_dim)
            self.stride_level_2=Upsample(c1_h,self.inter_dim,scale_factor=4)
        if level == 1:
            self.stride_level_0 =Conv(c1_l,self.inter_dim,2,2,0)
            self.stride_level_2 =Upsample(c1_h,self.inter_dim)
        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
    def forward(self,x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        w=self.weights_levels(levels_weight_v)
        w=F.softmax(w,dim=1)

        fused_out_reduced = level_0_resized * w[:, :1] + level_1_resized * w[:, 1:2] + level_2_resized * w[:, 2:3]
        return self.conv(fused_out_reduced)


class DenseASFF(nn.Module):
    def __init__(self, c1, c2, level=0):
        super(DenseASFF, self).__init__()

        # 定义输入的通道数
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = [c1_l, c1_m, c1_h]
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        # 定义不同层级的上采样和卷积操作
        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)
        elif level == 1:
            self.stride_level_0 = nn.Conv2d(c1_l, self.inter_dim, 3, 2, 1)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)
        elif level == 2:
            self.stride_level_0 = nn.Conv2d(c1_l, self.inter_dim, 3, 4, 1)
            self.stride_level_1 = nn.Conv2d(c1_m, self.inter_dim, 3, 2, 1)

        # 每个层级的特征权重
        self.weight_level_0 = nn.Conv2d(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = nn.Conv2d(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = nn.Conv2d(self.inter_dim, compress_c, 1, 1)

        # 密集连接部分，整合所有层级的特征图
        self.dense_conv = nn.Conv2d(self.inter_dim * 3, self.inter_dim, 1, 1)

        # 加权融合
        self.weights_levels = nn.Conv2d(compress_c * 3, 3, 1, 1)
        self.conv = nn.Conv2d(self.inter_dim, self.inter_dim, 3,1, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        # 根据当前层级调整输入特征图大小
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        # 计算权重并进行特征融合
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        # 连接所有权重向量并进行softmax归一化
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = F.softmax(self.weights_levels(levels_weight_v), dim=1)

        # 将所有层级的特征进行密集连接
        fused_feature = torch.cat((level_0_resized, level_1_resized, level_2_resized), 1)
        dense_fused = self.dense_conv(fused_feature)

        # 融合特征加权求和
        fused_out = dense_fused * levels_weight[:, :1] + \
                    dense_fused * levels_weight[:, 1:2] + \
                    dense_fused * levels_weight[:, 2:3]

        return self.conv(fused_out)

