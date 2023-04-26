_base_ = [
    # '../_base_/models/fcos_van_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py',    
    # '../_base_/schedules/schedule_4k.py',
    '../_base_/schedules/schedule_1x_adam.py',
]


dims = [64, 128, 320, 512]
# pretrained = dict(type='Pretrained', checkpoint='/home/nils/VAN-Detection/models/van-base_8xb128_in1k_20220501-conv.pth')
pretrained = None
norm_cfg = dict(type='SyncBN', requires_grad=not bool(pretrained))  # train BN only when not using pretrained model


model = dict(
    type='FCOS',
    backbone=dict(type='VAN', arch='b0', drop_path_rate=0.1, init_cfg=pretrained, 
                  out_indices=[0,1,2,3],
                 ),
    neck=dict(
        type='FPN',
        in_channels=dims,
        # in_channels=[512,],
        # in_channels=tuple(dims[-1]),
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
        dict(type='MMDetWandbHook', 
          by_epoch=True, 
          init_kwargs=dict(
            entity="nkoch-aitastic",
            project='van-detection', 
            tags=[
              'backbone:VAN-B0', 
              'neck:FPN',
              'head:FCOS', 
              # 'pretrained',
              'schedule:1x',
              ]       
          ),
          interval=10,
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        dict(type='TextLoggerHook', by_epoch=True),
    ])

  
data = dict(samples_per_gpu=1)
