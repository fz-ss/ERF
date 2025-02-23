# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: BN instead of SyncBN
# This work is licensed under the NVIDIA Source Code License
# A copy of the license is available at resources/license_segformer

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
# model = dict(
#     type='EncoderDecoder',
#     pretrained=None,
#     backbone=dict(type='IMTRv21_5', style='pytorch'),
#     decode_head=dict(
#         type='SegFormerHead',
#         in_channels=[64, 128, 320, 512],
#         in_index=[0, 1, 2, 3],
#         channels=128,
#         dropout_ratio=0.1,
#         num_classes=19,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         decoder_params=dict(),
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(768,768)))