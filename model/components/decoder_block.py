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
                 mlp_ratio,
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

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.act_layer = nn.GELU

        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=drop)

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            y: Input feature for cross attention, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        attention_mask = calculate_temporal_mask(self, x)

        shortcut = x
        x = self.norm1(x)

        x = self.cross_att_module(x, y, attention_mask)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
