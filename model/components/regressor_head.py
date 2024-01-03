import torch
import math
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn as nn


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def cnns(embed_dim, in_chans):
    # Define the CNN layers
    cnns = nn.Sequential(
        conv3x3(embed_dim * 8, embed_dim * 4),
        nn.GELU(),
        conv3x3(embed_dim * 4, embed_dim * 2),
        nn.GELU(),
        conv3x3(embed_dim * 2, embed_dim),
        nn.GELU(),
        conv3x3(embed_dim, in_chans),
    )

    return cnns


class regressor_head(nn.Module):
    def __init__(self,
                 embed_dim,
                 in_chans,
                 in_features,
                 out_features
                 ):
        super().__init__()
        self.cnns = cnns(embed_dim, in_chans)
        self.body_regressor_head = nn.Linear(in_features, out_features)

    def forward(self, x, H, W):

        features = x
        C = self.embed_dim * 8
        features = features.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()

        features = self.cnns(features)
        features = features.view(-1, self.in_chans, 13 * 12)
        out = self.body_regressor_head(features)


        return out
