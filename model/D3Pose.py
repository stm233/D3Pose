from components.encoder_block import *
from components.decoder_block import *
from components.util import *
from components.regressor_head import *


class D3Pose(nn.Module):
    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=2,
                 in_chans=10,
                 embed_dim=48,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.in_chans = in_chans

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # initialize encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderBlock(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=False)
            self.encoder_layers.append(layer)

        # initialize decoder layers
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DecoderBlock(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=False)
            self.Decoder_layers.append(layer)

        # initialize regression head
        self.CNNs = cnns(embed_dim, self.in_chans)
        self.body_regressor_head = nn.Linear(13 * 12, 85)

    # why init_weights
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, images, gt):
        """Forward function."""
        # images : batch x 10  x 200   x 192
        # x      : batch x 48*16 x size/2 x size/2
        # print(images.shape)

        # preprocess encoder input
        encoder_out = self.patch_embed(images)
        Wh_encoder, Ww_encoder = encoder_out.size(2), encoder_out.size(3)

        encoder_input = encoder_out.flatten(2).transpose(1, 2)
        encoder_input = self.pos_drop(encoder_input)

        # preprocess decoder input
        decoder_out = self.patch_embed(gt)
        Wh_decoder, Ww_decoder = encoder_input.size(2), encoder_input.size(3)

        for i in range(self.num_layers):
            encoder_block = self.encoder_layers[i]
            encoder_out, Wh_encoder, Ww_encoder = encoder_block(encoder_out, Wh_encoder, Ww_encoder)

            # encoder_out as part of decoder's input
            decoder_block = self.decoder_layers[i]
            decoder_out, Wh_decoder, Ww_decoder = decoder_block(decoder_out, encoder_out, Wh_decoder, Ww_decoder)

        # regress body parameters
        # y      : batch x 48*16*8 x size/16 x size/16
        features = decoder_out
        C = self.embed_dim * 8
        features = features.view(-1, Wh_decoder, Ww_decoder, C).permute(0, 3, 1, 2).contiguous()
        # y = y.view(-1, C // 4, Wh*2, Ww*2).contiguous()

        # z : batch x 81 x size/32 x size/32
        features = self.cnns(features)
        features = features.view(-1, self.in_chans, 13 * 12)
        out = self.body_regressor_head(features)
        # z = z.view(-1, self.in_chans, 15,3)

        return out

