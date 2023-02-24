_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
dims = [64, 128, 320, 512]
model = dict(
    type='ATSS',
    backbone=dict(
        type='VAN_Official',
        depths=[3, 3, 12, 3],
        drop_path_rate=0.2,
        embed_dims=dims,
        norm_cfg=norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint='/home/nils/VAN-Detection/models/van_base_828.pth.tar'),
    ),
    neck=[
        dict(
            type='FPN',
            in_channels=dims,
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True, reset_flag=True),
        dict(type='MMDetWandbHook', 
          by_epoch=True, 
          init_kwargs=dict(
            entity="nkoch-aitastic",
            project='van-detection', 
            tags=[
              'backbone:VAN-B2', 
              'neck:FPN',
              'neck:DyHead'
              'head:ATSS', 
              'pretrained',
              ]       
          ),
          interval=10,
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
    ])



# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
evaluation = dict(interval=1, metric=['bbox'])
