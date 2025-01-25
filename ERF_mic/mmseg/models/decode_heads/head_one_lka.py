# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule
import torchvision

class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # self.bottleneck = nn.Conv2d(( len(dilations) + int(pool) + int(bool(context_cfg))) * channels, channels, 3, 1)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)
# class DeformConv(nn.Module):

#     def __init__(self, in_channels, out_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
#         super(DeformConv, self).__init__()
        
#         self.offset_net = nn.Conv2d(in_channels=in_channels,
#                                     out_channels=2 * kernel_size[0] * kernel_size[1],
#                                     kernel_size=kernel_size,
#                                     padding=padding,
#                                     stride=stride,
#                                     dilation=dilation,
#                                     bias=True)

#         self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
#                                                         out_channels=out_channels,
#                                                         kernel_size=kernel_size,
#                                                         padding=padding,
#                                                         groups=groups,
#                                                         stride=stride,
#                                                         dilation=dilation,
#                                                         bias=False)

#     def forward(self, x):
#         offsets = self.offset_net(x)
#         out = self.deform_conv(x, offsets)
#         # print("deformable conv deformable conv deformable conv deformable conv deformable conv")
#         return out
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 深度卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 深度空洞卷积
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 逐点卷积
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        # 注意力操作
        return u * attn
# class Partial_conv3(nn.Module):

#     def __init__(self, dim, n_div):
#         super().__init__()
#         self.dim_conv3 = dim // n_div
#         self.dim_untouched = dim - self.dim_conv3
#         self.partial_conv3 = nn.Conv2d( self.dim_conv3,  self.dim_conv3, 5, padding=2, groups=self.dim_conv3)
#         self.conv_spatial = nn.Conv2d(
#             self.dim_untouched,   self.dim_untouched, 7, stride=1, padding=9, groups=self.dim_untouched, dilation=3)
#         self.forward = self.forward_split_cat

#     def forward_split_cat(self, x):
#         # for training/inference
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
#         x1 = self.partial_conv3(x1)
#         x2 = self.conv_spatial(x2)
#         x = torch.cat((x1, x2), 1)

#         return x
# class LKA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # 深度卷积
#         self.conv0 = Partial_conv3(dim,2)

#             # nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

#         # 深度空洞卷积
#         # self.conv_spatial = nn.Conv2d(
#         #     dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
#         # 逐点卷积
#         self.conv1 = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         # attn = self.conv_spatial(attn)
#         attn = self.conv1(attn)

        # 注意力操作
        return u * attn
@HEADS.register_module()
class Headone(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(Headone, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        # self.lka0 = LKA(self.channels)
        # self.lka1 = LKA(self.channels)
        # self.lka2 = LKA(self.channels)
        # self.lka3 = LKA(self.channels)

        self.embed_layers = nn.ModuleDict(self.embed_layers)
        # self.conv = ConvModule(
        #     sum(embed_dims),
        #     self.channels,
        #     kernel_size=3,
        #     padding=1,
        #     norm_cfg=fusion_cfg["norm_cfg"],
        #     act_cfg=fusion_cfg["act_cfg"],
        #     groups=self.channels)
        self.conv =  nn.Conv2d(sum(embed_dims), self.channels, 3,1,1,groups=self.channels)
        # self.dconv = DeformConv(sum(embed_dims), self.channels, kernel_size=(3,3), padding=1, groups=self.channels)
        # self.relu = nn.ReLU()
        self.lka = LKA(self.channels)


        self.fuse_layer = build_layer(
            self.channels, self.channels, **fusion_cfg)

    def forward(self, inputs):
        x = inputs


        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners) #2,256,
        # print(_c[0].shape)
        # print(_c[1].shape)
        # print(_c[2].shape)
        # print(_c[3].shape)
        # exit()

        # _c[0] = self.lka0(_c[0])
        # _c[1] = self.lka0(_c[1])
        # _c[2] = self.lka0(_c[2])
        # _c[3] = self.lka0(_c[3])
        x = self.conv(torch.cat(list(_c.values()), dim=1))
        x = self.lka(x)
        # x = self.dconv(torch.cat(list(_c.values()), dim=1))

        x = self.fuse_layer(x)
        x = self.cls_seg(x)

        return x
