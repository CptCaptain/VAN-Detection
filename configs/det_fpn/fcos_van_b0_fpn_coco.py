_base_ = [
    '../_base_/models/fcos_van_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py',    
    '../_base_/schedules/schedule_4k.py',
]

dims = [32, 64, 160, 256]
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
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
        max_per_img=100)
        )


log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MMDetWandbHook', 
          by_epoch=False, 
          init_kwargs=dict(
            entity="nkoch-aitastic",
            project='van-detection', 
            tags=[
              'backbone:VAN-B0', 
              'neck:FPN',
              'head:FCOS', 
              'pretrained',
              ]       
          ),
          interval=10,
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
    ])

  
data = dict(samples_per_gpu=4)
