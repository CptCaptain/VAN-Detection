custom_imports = dict(imports=['finetunecopypaste'], allow_failed_imports=False)

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               # 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
               'vase', 'scissors', 'teddy bear', 'Audi_A7', 'Audi_RS_6_Avant')

dataset_type = 'CocoDataset'
data_root = '/home/nils/VAN-Detection/datasets/coco/'

image_size = (1024, 1024)
file_client_args = dict(backend='disk')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
                    skip_img_without_anno=True,
                ),
            ],
            classes=CLASSES,
        ),
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

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
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            classes=CLASSES,
            img_prefix=data_root + 'train2017/',
            pipeline=load_pipeline),
        pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file='/home/nils/datasets/cars/val_data/val.json',
        img_prefix='/home/nils/datasets/cars/val_data/img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file='/home/nils/datasets/cars/val_data/val.json',
        img_prefix='/home/nils/datasets/cars/val_data/img/',
        pipeline=test_pipeline))
