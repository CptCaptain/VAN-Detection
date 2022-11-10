_base_ = [
    '../_base_/default_runtime.py',    
    '../_base_/schedules/schedule_1x_adam.py',
]

dims = [64, 128, 320, 512]
depths = [3, 3, 12, 3]
pretrained = None
norm_cfg = dict(type='SyncBN', requires_grad=not bool(pretrained))  # train BN only when not using pretrained model


model = dict(
    type='FCOS',
    backbone=dict(
        type='VAN',
        embed_dims=dims,
        drop_rate=0.0,
        drop_path_rate=0.2,
        depths=depths,
        norm_cfg=norm_cfg,
        init_cfg=pretrained,
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
    interval=10,
    hooks=[
        dict(type='MMDetWandbHook', 
          by_epoch=True, 
          init_kwargs=dict(
            entity="nkoch-aitastic",
            project='van-detection', 
            tags=[
              'backbone:VAN-B2', 
              'neck:FPN',
              'head:FCOS', 
              # 'pretrained',
              'schedule:1x-adam',
              'standard-scale-jitter',
              'copy-paste',
              ]       
          ),
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        dict(type='TextLoggerHook', by_epoch=True),
    ])

  
# data = dict(samples_per_gpu=4)


# Straight up taken from yolox config to see if this can run for more than 1 epoch

# dataset settings
data_root = 'datasets/coco/'
dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# image_size = (1333, 800)
image_size = (1024, 1024)
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='Resize',
                img_scale=image_size,
                ratio_range=(0.8, 1.25),
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=image_size,
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Pad', size=image_size),
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

workflow=[('train', 12), ('val', 1)]
