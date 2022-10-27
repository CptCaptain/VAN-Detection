# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model_base = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VAN',
        embed_dims=[32, 64, 160, 256],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=norm_cfg),
    neck=dict(
        type='FPN',
        # in_channels=[32, 64, 160, 256],
        out_channels=256,
        num_outs=4
        ),
    bbox_head=dict(
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
          ),
    # model training and testing settings
    train_cfg=dict(
      assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100)
    )
dims = [32, 64, 160, 256]
model_retina = dict(
    type='RetinaNet',
    backbone=dict(
        type='VAN',
        embed_dims=dims,
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint='/content/models/van_tiny_754.pth.tar'),
      ),
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=dims,
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
)


model_fcos = dict(
    type='FCOS',
    backbone=dict(
        type='VAN',
        embed_dims=dims,
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint='/content/models/van_tiny_754.pth.tar'),
      ),
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='caffe',
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=dims,
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
        
model = model_fcos
