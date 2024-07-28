# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
# pd.options.plotting.backend = "plotly"
from glob import glob

import pandas as pd
import torch.nn.functional as F
import torch.nn.functional as Functional
from scipy import ndimage
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from tqdm import tqdm

import copy
# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch.backends.cudnn as cudnn
# Sklearn

# PyTorch
from torch.utils.data import Dataset

# Albumentations for augmentations
import albumentations as A

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss as smpDL, SoftBCEWithLogitsLoss, TverskyLoss

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import ml_collections
DATA_PATH = 'D:/kaggle'
tqdm.pandas()
wf_seed = 0


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list


def worker_init_fn(worker_id):
    random.seed(worker_id)


# import math


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        # v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        v = torch.var(w, dim=[1, 2, 3], unbiased=False, keepdim=True)
        m = torch.mean(w, dim=[1, 2, 3], keepdim=True)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[(n_block + "/" + n_unit + "/" + "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[(n_block + "/" + n_unit + "/" + "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[(n_block + "/" + n_unit + "/" + "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[(n_block + "/" + n_unit + "/" + "gn1/scale")])
        gn1_bias = np2th(weights[(n_block + "/" + n_unit + "/" + "gn1/bias")])

        gn2_weight = np2th(weights[(n_block + "/" + n_unit + "/" + "gn2/scale")])
        gn2_bias = np2th(weights[(n_block + "/" + n_unit + "/" + "gn2/bias")])

        gn3_weight = np2th(weights[(n_block + "/" + n_unit + "/" + "gn3/scale")])
        gn3_bias = np2th(weights[(n_block + "/" + n_unit + "/" + "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[(n_block + "/" + n_unit + "/" + "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[(n_block + "/" + n_unit + "/" + "gn_proj/scale")])
            proj_gn_bias = np2th(weights[(n_block + "/" + n_unit + "/" + "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert 3 > pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


"""
This version of transunet contains the TUP and BiF block as implemented by me according to the equations in 
the paper.
Saved in separate file before transformer decoder integration.

"""
logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


####### TUP ########
class StdConv2d(nn.Conv2d):
    """Standard Convolution with weight normalization."""

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return Functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DoubleConv(nn.Module):
    """(convolution(3x3, pad=1) => [BN] => ReLU) * 2
    Currently uses the wrapper above"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            StdConv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            StdConv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class TUP(nn.Module):
    """Transformer UP-sampler.
    Inputs: transformer output features
    Outputs: transformer output features scaled up, in preparation to enter BiFusion block.
    """

    def __init__(self, cin, cout):
        super().__init__()
        cmid = (cin + cout) // 2
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dblConv = DoubleConv(cin, cout, cmid)
        self.conv1 = nn.Sequential(
            StdConv2d(cin, cout, kernel_size=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Upsample features
        upsampled_x = self.up(x)

        # Residual branch processing
        residual = self.conv1(upsampled_x)

        # Main branch processing
        y = self.dblConv(upsampled_x)

        # Integrate features through element-wise multiplication
        y = y * residual

        return y


####### BiFusion ########
"""
This variant is loyal to the implementation mentioned in the sources provided in BiFTransNet
"""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class AvgSpatial(nn.Module):
    def forward(self, x):
        return Functional.avg_pool2d(x, x.size()[2:])


class TripleConv(nn.Module):
    def __init__(self, num_channels):
        super(TripleConv, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        do_bn = x.size(2) > 1 and x.size(3) > 1  # Check if spatial dimensions are larger than 1x1
        identity = x

        out = self.conv1(x)
        if do_bn:
            out = self.bn1(out)
        out = self.relu1(out)

        out = out + identity  # Skip connection

        out = self.conv2(out)
        if do_bn:  # Check again for spatial dimensions
            out = self.bn2(out)
        out = self.relu2(out)

        out = out + identity  # Another skip connection

        out = self.conv3(out)
        if do_bn:  # Final check for spatial dimensions
            out = self.bn3(out)
        out = self.relu3(out)

        return out


class ConvChannelAttention(nn.Module):
    def __init__(self, num_channels):
        super(ConvChannelAttention, self).__init__()
        self.avg_spatial = AvgSpatial()  # Average spatial pooling
        self.triple_conv = TripleConv(num_channels)  # Triple convolution block
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

    def forward(self, x):
        out = self.avg_spatial(x)  # Eq. (8)
        out = self.triple_conv(out)  # Eq. (9)
        out = self.sigmoid(out) * x  # Eq. (10)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.InstanceNorm2d(1, affine=True)  # in lieu of batchnorm
        # fixes the following error:
        # RuntimeError: Function NativeBatchNormBackward0 returned an invalid gradient at index 1 -
        # got [] but expected shape compatible with [1]

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class MultimodalFusion(nn.Module):
    def __init__(self, num_channels):
        super(MultimodalFusion, self).__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

    def forward(self, ti, ci):
        # Adjusting to match the paper's description -
        # early version contained a mistake (concatenation instead of Hadamard)
        hadamard_product = ti * ci  # Element-wise multiplication (Hadamard product)
        fused = self.conv(hadamard_product)  # Apply convolution
        return fused


class BiFusion_block(nn.Module):
    def __init__(self, num_channels, use_conv_channel_att=True):
        super(BiFusion_block, self).__init__()
        if use_conv_channel_att:
            print("Initialized conv channel attention!")
            self.ca = ConvChannelAttention(num_channels)
        else:
            self.ca = ChannelAttention(num_channels)  # Channel Attention
        self.sa = SpatialAttention()  # Spatial Attention
        self.fusion = MultimodalFusion(num_channels)  # Multimodal Fusion
        self.residual = nn.Sequential(
            nn.Conv2d(num_channels * 3, num_channels, kernel_size=1),  # Residual connection
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ti, ci):
        t3 = self.ca(ti)  # Output of Channel Attention
        c5 = self.sa(ci)  # Output of Spatial Attention
        fi = self.fusion(ti, ci)  # Output of Multimodal Fusion
        concat = torch.cat([t3, c5, fi], dim=1)  # Concatenate outputs
        out = self.residual(concat)  # Pass through residual module
        return out


### ### ###


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** (1 / 2))
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[(ROOT + "/" + ATTENTION_Q + "/" + "kernel.npy")]).view(self.hidden_size,
                                                                                                self.hidden_size).t()
            key_weight = np2th(weights[(ROOT + "/" + ATTENTION_K + "/" + "kernel.npy")]).view(self.hidden_size,
                                                                                              self.hidden_size).t()
            value_weight = np2th(weights[(ROOT + "/" + ATTENTION_V + "/" + "kernel.npy")]).view(self.hidden_size,
                                                                                                self.hidden_size).t()
            out_weight = np2th(weights[(ROOT + "/" + ATTENTION_OUT + "/" + "kernel.npy")]).view(self.hidden_size,
                                                                                                self.hidden_size).t()

            query_bias = np2th(weights[(ROOT + "/" + ATTENTION_Q + "/" + "bias.npy")]).view(-1)
            key_bias = np2th(weights[(ROOT + "/" + ATTENTION_K + "/" + "bias.npy")]).view(-1)
            value_bias = np2th(weights[(ROOT + "/" + ATTENTION_V + "/" + "bias.npy")]).view(-1)
            out_bias = np2th(weights[(ROOT + "/" + ATTENTION_OUT + "/" + "bias.npy")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[(ROOT + "/" + FC_0 + "/" + "kernel.npy")]).t()
            mlp_weight_1 = np2th(weights[(ROOT + "/" + FC_1 + "/" + "kernel.npy")]).t()
            mlp_bias_0 = np2th(weights[(ROOT + "/" + FC_0 + "/" + "bias.npy")]).t()
            mlp_bias_1 = np2th(weights[(ROOT + "/" + FC_1 + "/" + "bias.npy")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[(ROOT + "/" + ATTENTION_NORM + "/" + "scale.npy")]))
            self.attention_norm.bias.copy_(np2th(weights[(ROOT + "/" + ATTENTION_NORM + "/" + "bias.npy")]))
            self.ffn_norm.weight.copy_(np2th(weights[(ROOT + "/" + MLP_NORM + "/" + "scale.npy")]))
            self.ffn_norm.bias.copy_(np2th(weights[(ROOT + "/" + MLP_NORM + "/" + "bias.npy")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
            use_bi_fusion=True,
    ):
        super().__init__()
        """
        BiF block here!
        The bif block is a per-tier block - instead of in-channels + skip-channels as inputs,
        we want to get transformer-upsampled channels + skip-channels.
        """
        self.use_bi_fusion = False
        if use_bi_fusion and skip_channels:
            self.use_bi_fusion = True
            self.bifusion = BiFusion_block(skip_channels)

        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None, tup_out=None):
        x = self.up(x)
        if skip is not None:
            """
            Forward pass addition of the BiF block
            """
            if self.use_bi_fusion and tup_out is not None:
                assert skip.shape == tup_out.shape
                skip = self.bifusion(skip, tup_out)

            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        """
        Insert TUP here - each skip connection input to a decoder block needs a dedicated TUP component 
        """
        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        tup_blocks = None
        if skip_channels:
            tup_blocks = []
            # first tup - take head and set it to skip connection size
            last_tup_input_size = head_channels
            for idx in range(len(skip_channels)):

                if skip_channels[idx]:  # we actually use this skip channel
                    # then create a tup block and set input=last tup output and output = skip channel size
                    # print(f"{skip_channels[idx]=}, {idx=}, {last_tup_input_size=}")
                    tup_blocks.append(TUP(last_tup_input_size, skip_channels[idx]))
                    last_tup_input_size = skip_channels[idx]

        self.blocks = nn.ModuleList(blocks)
        if tup_blocks is not None:
            # define TUP blocks modules
            self.TUP_blocks = nn.ModuleList(tup_blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)  # x is now basically the transformer output
        tup_out = x  # save the original state of transformer embeddings
        tup_idx = 0
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
                if tup_idx < len(self.TUP_blocks):
                    # use TUP:
                    tup_out = self.TUP_blocks[tup_idx](tup_out)
                    tup_idx += 1
            else:
                skip = None
            x = decoder_block(x, skip=skip, tup_out=tup_out)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, use_2p5D=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=3 * config['n_classes'] if use_2p5D else config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        self.TwoPointFiveD = use_2p5D

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        if self.TwoPointFiveD:
            logits = logits.view(logits.size(0), 3, logits.size(1) // 3, logits.size(2), logits.size(3))
        return logits

    def load_from(self, weights):

        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel.npy"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias.npy"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale.npy"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias.npy"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding.npy"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel.npy"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale.npy"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias.npy"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-L_32': get_l32_config(),
    'ViT-H_14': get_h14_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
    'R50-ViT-L_16': get_r50_l16_config(),
    'testing': get_testing(),
}


# For colored terminal text

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def id2mask(id_, df):
    idf = df[df['id'] == id_]
    wh = idf[['height', 'width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(['large_bowel', 'small_bowel', 'stomach']):
        cdf = idf[idf['class'] == class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask


def load_img(path):
    img = np.load(path)
    img = img.astype('float32')  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk /= 255.0
    return msk


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     img = clahe.apply(img)
    #     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis('off')


# build dataset
class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


def apply_gamma_correction(image, **kwargs):
    """Applies gamma correction to an image."""
    # Ensure the image is within a valid range (assuming 8-bit images)
    image = np.clip(image, 0, 255)

    gamma = np.random.uniform(0.9, 1.1)

    # Apply gamma correction
    corrected_image = image ** gamma

    return corrected_image


data_transforms = {
    "train": A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
        # A.Flip(p=0.5),  # Flip
        # A.Rotate(limit=(-90, 90), p=0.5),  # Rotation
        # A.OneOf([
        #     A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
        # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
        # A.GaussNoise(var_limit=0.1, mean=0, per_channel=True, p=1.),  # White Gaussian Noise
        # A.GaussianBlur(blur_limit=(3, 7), p=1.),  # Gaussian Blurring
        # A.Lambda(image=apply_gamma_correction, p=1.),  # Gamma self-implemented
        # A.RandomBrightnessContrast(p=1.),  # Adjusting Brightness and Contrast

        # ], p=0.25),
        # A.CoarseDropout(max_holes=8, max_height=224, max_width=224, min_holes=5,
        #                 fill_value=0, mask_fill_value=0, p=0.5),  # Coarse Dropout
        # A.Downscale(scale_min=0.75, scale_max=0.9, p=0.5),  # Simulation of Low Resolution
        # A.RandomGamma(always_apply=False, p=0.5)  # Gamma transformation (causes issues???)
    ], p=1.0),

    "valid": A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
    ], p=1.0)
}


# dataloader
def prepare_loaders(df, debug=False):
    train_df = df  # df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df  # df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32 * 5).query("empty==0")
        valid_df = valid_df.head(32 * 3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    return train_dataset, valid_dataset


def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(3 * 3, 3))
    for idx in range(size):
        plt.subplot(1, 3, idx + 1)
        img = imgs[idx,].permute((1, 2, 0)).numpy() * 255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy() * 255.0
        show_img(img, msk)
    plt.tight_layout()
    plt.show()


def main_2p5_D(seed, bs, vs):
    # DATA
    path_df = pd.DataFrame(glob(DATA_PATH+'/input/uwmgi-25d-stride2-dataset/images/images/*'), columns=['image_path'])
    path_df['mask_path'] = path_df.image_path.str.replace('image', 'mask')
    path_df['id'] = path_df.image_path.map(lambda x: x.split('\\')[-1].replace('.npy', ''))
    # print(path_df.head())

    df = pd.read_csv(DATA_PATH+'/input/uwmgi-mask-dataset/train.csv')

    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len)  # length of each rle mask

    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()  # rle list of each id
    df2 = df2.merge(
        df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())  # total length of all rles of each id

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df['empty'] = (df.rle_len == 0)  # empty masks
    # print(df.head())

    df = df.drop(columns=['image_path', 'mask_path'])

    df = df.merge(path_df, on=['id'])

    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)
    df.head()

    # Prepare data loaders
    train_loader, valid_loader = prepare_loaders(df)
    return train_loader


def trainer_uw(args, model, snapshot_path, device):
    global wf_seed
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    wf_seed = random.seed()
    db_train = main_2p5_D(wf_seed, args.batch_size, args.batch_size)
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, pin_memory=True,
                             worker_init_fn=worker_init_fn, num_workers=8)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    bce_loss = SoftBCEWithLogitsLoss()
    dice_s = smpDL(mode='multilabel')
    tversky = TverskyLoss(mode='multilabel', log_loss=False)
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # writer = SummaryWriter(snapshot_path + '/log')
    with SummaryWriter(snapshot_path + '/log') as writer:
        iter_num = 0
        max_epoch = args.max_epochs
        max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
        best_performance = 0.0
        iterator = tqdm(range(max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, (_img, _msk) in enumerate(trainloader):
                image_batch, label_batch = _img, _msk
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                outputs = model(image_batch)
                loss_bce = bce_loss(outputs, label_batch)
                loss = 0.5 * loss_bce + 0.5 * tversky(outputs, label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

                logging.info('iteration %d : loss : %f, loss_bce: %f' % (iter_num, loss.item(), loss_bce.item()))

                if iter_num == 1 or iter_num % 20 == 0:
                    # This segement used to store images with masks
                    # Does not fit 2.5D training?
                    # pass
                    image = image_batch[1, :, :, :].view(3, 224, 224)
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].view(3, 224, 224)
                    writer.add_image('train/GroundTruth', labs, iter_num)

            save_interval = 50  # int(max_epoch/6)
            if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if epoch_num >= max_epoch - 1:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                iterator.close()
                break

    # writer.close()
    return "Training Finished!"


# AMD stuff
import torch_directml

dml = torch_directml.device()

"""
Rotem: modified the args: epoch set to 50 (original = 150)
        modified batch size to 16 (original 24) and lr decreased linearly according to batch size (16/24 * original_lr)
"""

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='UW-Madison', help='experiment_name')
# parser.add_argument('--dataset', type=str,
#                     default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=(16 / 24) * 0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '..\\data\\Synapse\\train_npz',
            'list_dir': '.\\lists\\lists_Synapse',
            'num_classes': 9,
        },
        'UW-Madison': {
            'root_path': '..\\data\\Synapse\\train_npz',
            'list_dir': '.\\lists\\lists_Synapse',
            'num_classes': 3,
        },
    }
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "..\\model\\{}\\{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    trainer = {'UW-Madison': trainer_uw}

    net = VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(dml)
    net.load_from(weights=np.load(config_vit.pretrained_path))
    torch.autograd.set_detect_anomaly(True)
    trainer[dataset_name](args, net, snapshot_path, dml)
