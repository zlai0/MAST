import torch
import torch.nn as nn

import pdb
from .submodule import ResNet18
from .colorizer import Colorizer

import numpy as np

class MAST(nn.Module):
    def __init__(self, args):
        super(MAST, self).__init__()

        # Model options
        self.p = 0.3
        self.C = 7
        self.args = args

        self.feature_extraction = ResNet18(3)
        self.post_convolution = nn.Conv2d(256, 64, 3, 1, 1)
        self.D = 4

        # Use smaller R for faster training
        if args.training:
            self.R = 6
        else:
            self.R = 12

        self.colorizer = Colorizer(self.D, self.R, self.C)

    def forward(self, rgb_r, quantized_r, rgb_t, ref_index=None,current_ind=None):
        feats_r = [self.post_convolution(self.feature_extraction(rgb)) for rgb in rgb_r]
        feats_t = self.post_convolution(self.feature_extraction(rgb_t))

        quantized_t = self.colorizer(feats_r, feats_t, quantized_r, ref_index, current_ind)
        return quantized_t


    def dropout2d_lab(self, arr): # drop same layers for all images
        if not self.training:
            return arr

        drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
        drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

        for a in arr:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))

        return arr, drop_ch_ind # return channels not masked