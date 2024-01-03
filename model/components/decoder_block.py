import torch
import math
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from att_module import *
from swin_blocks import *
from util import *


# calculate temporal mask
def calculate_temporal_mask(self, trg):
    N, trg_len = trg.shape
    trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        N, 1, trg_len, trg_len
    )

    return trg_mask.to(self.device)


class DecoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 inverse=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.cross_att_module = cross_attention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop
        )

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            y: Input feature for cross attention, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        attention_mask = calculate_temporal_mask(self, x)

        x = self.cross_att_module(x, y, attention_mask)
        return x
