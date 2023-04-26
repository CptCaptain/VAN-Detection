_base_ = [
    '../_base_/models/fcos_van_fpn.py',
    '../_base_/datasets/finetune_coco_detection.py',
    '../_base_/default_runtime.py',    
    '../_base_/schedules/schedule_1x_adam.py',
]

data_root = '/home/nils/VAN-Detection/datasets/coco/'
CLASSES = ('Audi_A7', 'Audi_RS_6_Avant')

dims = [32, 64, 160, 256]
# pretrained = dict(type='Pretrained', checkpoint='models/van_b0_bt_300.pth', prefix='backbone.')
# pretrained = dict(type='Pretrained', checkpoint='models/van_b0_bt_small_neck_1000.pth', prefix='backbone.')
pretrained = dict(type='Pretrained', checkpoint='models/van_b0_bt_thick_neck_1000.pth', prefix='backbone.')
# pretrained = None
norm_cfg = dict(type='SyncBN', requires_grad=not bool(pretrained))  # train BN only when not using pretrained model
image_size = (1024, 1024)
# image_size = (1333, 800)
file_client_args = dict(backend='disk')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='FCOS',
    backbone=dict(
        type='VAN',
        _delete_=True,
        arch='b0',
        out_indices=[0,1,2,3],
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
        num_classes=len(CLASSES),
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


hooks=[
    dict(type='TextLoggerHook', by_epoch=True),
    dict(type='MMDetWandbHook', 
      by_epoch=True, 
      init_kwargs=dict(
        entity="nkoch-aitastic",
        project='van-detection', 
        tags=[
          'backbone:VAN-B0', 
          'neck:FPN',
          'head:FCOS', 
          'pretrained',
          'schedule:1x',
          'finetune-copy-paste',
          'barlow-twins:stanford-cars',
          'less-classes',
          ]       
      ),
      interval=10,
      log_checkpoint=True,
      log_checkpoint_metadata=True,
      num_eval_images=100,
    ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
    ]

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='Resize',
                width=image_size[1],
                height=image_size[0],
            ),
            dict(
                type='RandomResizedCrop',
                width=image_size[1],
                height=image_size[0],
            ),
        ],
        p=1.0
    ),
    dict(
        type='ShiftScaleRotate',
        scale_limit=[-0.9,0.0],
        rotate_limit=0,
        border_mode=0,
        rotate_method='ellipse',
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=128,
                sat_shift_limit=128,
                val_shift_limit=128,
                p=1.0),
        ],
        p=0.9),
    dict(type='ImageCompression', quality_lower=45, quality_upper=95, p=0.3),
    dict(type='ChannelShuffle', p=0.4),
    dict(type='GaussNoise', p=0.8),
    dict(
        type='OneOf',
        transforms=[
            dict(type='AdvancedBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
        ],
        p=0.4),
    dict(
        type='ColorJitter',
        ),
    dict(
        type='RandomCrop',
        width=400,
        height=400,
        p=0.2,
        ),
    dict(
        type='Flip',
        ),
    dict(
        type='Resize',
        width=image_size[1],
        height=image_size[0],
        ),
    dict(
        type='PadIfNeeded',
        min_width=image_size[1],
        min_height=image_size[0],
        border_mode=0,
        p=1.0,
        ),
]

test_pipeline = [
    dict(
        type='FineTuneCopyPaste', 
        max_num_pasted=100,
        copy_paste_chance=0.8,
        supl_dataset_cfg=dict(
            ann_file='/home/nils/datasets/cars/coco/train.json',
            data_root='/home/nils/datasets/cars/',
            img_prefix='raw',
            pipeline=[
                dict(type='LoadImageFromFile', file_client_args=file_client_args),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Albu',
                    transforms=albu_train_transforms,
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_labels'],
                        min_visibility=0.0,
                        filter_lost_elements=True,
                    ),
                    update_pad_shape=False,
                    skip_img_without_anno=False,
                ),
            ],
            classes=CLASSES,
        ),
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

train_pipeline = [
    dict(
        type='FineTuneCopyPaste', 
        max_num_pasted=100,
        copy_paste_chance=0.8,
        supl_dataset_cfg=dict(
            ann_file='/home/nils/datasets/cars/coco/train.json',
            data_root='/home/nils/datasets/cars/',
            img_prefix='raw',
            pipeline=[
                dict(type='LoadImageFromFile', file_client_args=file_client_args),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Albu',
                    transforms=albu_train_transforms,
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_labels'],
                        min_visibility=0.0,
                        filter_lost_elements=True,
                    ),
                    update_pad_shape=False,
                    skip_img_without_anno=False,
                ),
            ],
            classes=CLASSES,
        ),
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

load_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        # ratio_range=(0.8, 1.25),
        # multiscale_mode='range',
        keep_ratio=False,
        ),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Pad', size_divisor=32),
    dict(type='Pad', size=image_size),
]

data = dict(
    samples_per_gpu=4,
    train=dict(
        dataset=dict(
            classes=CLASSES,
            filter_empty_gt=False,
        ),
        pipeline=train_pipeline,
    ),
    val=dict(
        classes=CLASSES,
    ),
    test=dict(
        classes=CLASSES,
    ),
)

    # hooks = [
    # dict(type='TextLoggerHook', by_epoch=True),
# ]
log_config = dict(
    hooks=hooks,
)

# # FIXME This crashes right now due to incorrect mapping of classes during evaluation
# # We skip this hook until we're done training, so we at least get something done...
# evaluation = dict(interval=12)
# checkpoint_config = dict(interval=12)

workflow = [('train', 1)]
# data = dict(samples_per_gpu=4)

