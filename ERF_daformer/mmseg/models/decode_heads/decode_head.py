# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Support for seg_weight
# - Add debug flag

from abc import ABCMeta, abstractmethod
import cv2
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.models.utils.visualization import Cityscapes_palette
import numpy as np
import PIL

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
def colorize_mask(mask, palette): #512,512
    zero_pad = 256*3 -len(palette) #0
    for i in range(zero_pad):
        palette.append(0)
    new_mask=[]
    new_mask0 = PIL.Image.fromarray(mask[0].astype(np.uint8)).convert('P')
    new_mask0.putpalette(palette)
    new_mask0 = new_mask0.convert('RGB')
    new_mask.append(np.array(new_mask0))
    new_mask1 = PIL.Image.fromarray(mask[1].astype(np.uint8)).convert('P')
    new_mask1.putpalette(palette)
    new_mask1 = new_mask1.convert('RGB')
    new_mask.append(np.array(new_mask1))
    return new_mask

class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        self.debug = False
        self.debug_output = None

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        # ############################################################################################################# Sobel loss 

        _, seg_logit = torch.max(seg_logit, 1) #2,512,512
        seg_logit = seg_logit.data.cpu()
        seg_logit = seg_logit.squeeze(0)
        if torch.is_tensor(seg_logit):
            seg_logit = seg_logit.numpy() #2,512,512
        color_src = seg_label.data.cpu()
        if torch.is_tensor(color_src):
            color_src = color_src.numpy() #2,512,512

        color = colorize_mask(seg_logit, Cityscapes_palette) #512,512
        color_src = colorize_mask(color_src, Cityscapes_palette) #512,512
        # color = cv2.imread(color)
        # print(color.shape)
        # exit()
        color = np.array(color) 
        color_src = np.array(color_src) 
        
        # color.save("/home/ailab/4T/SYN/HRDA/img/a.png")
        # img_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)/255.
        ####彩色圖
        img_gray = []
        img_gray.append(cv2.cvtColor(color[0], cv2.COLOR_BGR2GRAY)/255.)
        img_gray.append(cv2.cvtColor(color[1], cv2.COLOR_BGR2GRAY)/255.)
        color = np.array(img_gray)
        del img_gray
        img_gray_src = []
        img_gray_src.append(cv2.cvtColor(color_src[0], cv2.COLOR_BGR2GRAY)/255.)
        img_gray_src.append(cv2.cvtColor(color_src[1], cv2.COLOR_BGR2GRAY)/255.)
        color_src = np.array(img_gray_src)
        del img_gray_src
        ####彩色圖 over


        ####原來的class picture
        # color = color/19.   ###no rgb image
    
        sobelx = cv2.Sobel(src=color, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobely = cv2.Sobel(src=color, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobelxy = cv2.Sobel(src=color, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
        
        sub_y_x = torch.tensor(sobely - sobelx).cuda()
        sub_xy_x = torch.tensor(sobelxy - sobelx).cuda()
        sub_xy_y = torch.tensor(sobelxy - sobely).cuda()

        sobelx_src = cv2.Sobel(src=color_src, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobely_src = cv2.Sobel(src=color_src, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobelxy_src = cv2.Sobel(src=color_src, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
        
        sub_y_x_src = torch.tensor(sobely_src - sobelx_src).cuda()
        sub_xy_x_src = torch.tensor(sobelxy_src - sobelx_src).cuda()
        sub_xy_y_src = torch.tensor(sobelxy_src - sobely_src).cuda()

        loss_s = torch.nn.L1Loss()
        theta = 0.3
        # zero_img = torch.zeros(seg_label.shape[0], seg_label.shape[1], seg_label.shape[2]).cuda()
        loss['loss_y_x'] = theta*loss_s(sub_y_x, sub_y_x_src)
        loss['loss_xy_x'] = theta*loss_s(sub_xy_x, sub_xy_x_src)
        loss['loss_xy_y'] = theta*loss_s(sub_xy_y, sub_xy_y_src)
        ############################################################################################################## Sobel loss over
        return loss
