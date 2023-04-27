data = {'val': {'type': 'CocoDataset', 'classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'Audi_RS_6_Avant'], 'ann_file': 'datasets/coco/annotations/instances_val2017.json', 'pipeline': [{'type': 'LoadImageFromFile'}, {'flip': False, 'type': 'MultiScaleFlipAug', 'img_scale': [(1333, 800)], 'transforms': [{'type': 'Resize', 'keep_ratio': True}, {'type': 'RandomFlip'}, {'std': [58.395, 57.12, 57.375], 'mean': [123.675, 116.28, 103.53], 'type': 'Normalize', 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'keys': ['img'], 'type': 'ImageToTensor'}, {'keys': ['img'], 'type': 'Collect'}]}], 'img_prefix': 'datasets/coco/val2017/'}, 'test': {'type': 'CocoDataset', 'classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'Audi_RS_6_Avant'], 'ann_file': 'datasets/coco/annotations/instances_val2017.json', 'pipeline': [{'type': 'LoadImageFromFile'}, {'flip': False, 'type': 'MultiScaleFlipAug', 'img_scale': [(1333, 800)], 'transforms': [{'type': 'Resize', 'keep_ratio': True}, {'type': 'RandomFlip'}, {'std': [58.395, 57.12, 57.375], 'mean': [123.675, 116.28, 103.53], 'type': 'Normalize', 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'keys': ['img'], 'type': 'ImageToTensor'}, {'keys': ['img'], 'type': 'Collect'}]}], 'img_prefix': 'datasets/coco/val2017/'}, 'train': {'type': 'MultiImageMixDataset', 'dataset': {'type': 'CocoDataset', 'classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'Audi_RS_6_Avant'], 'ann_file': 'datasets/coco/annotations/instances_train2017.json', 'pipeline': [{'type': 'LoadImageFromFile', 'file_client_args': {'backend': 'disk'}}, {'type': 'LoadAnnotations', 'with_bbox': True, 'with_mask': True}, {'type': 'Resize', 'img_scale': [(1024, 1024)], 'keep_ratio': True, 'ratio_range': [0.8, 1.25], 'multiscale_mode': 'range'}, {'type': 'RandomCrop', 'crop_size': [1024, 1024], 'crop_type': 'absolute_range', 'recompute_bbox': True, 'allow_negative_crop': True}, {'type': 'FilterAnnotations', 'min_gt_bbox_wh': [0.01, 0.01]}, {'type': 'RandomFlip', 'flip_ratio': 0.5}, {'size': [1024, 1024], 'type': 'Pad'}], 'img_prefix': 'datasets/coco/train2017/'}, 'pipeline': [{'type': 'FineTuneCopyPaste', 'max_num_pasted': 100, 'supl_dataset_cfg': {'classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'Audi_RS_6_Avant'], 'ann_file': '/home/nils/datasets/cars/coco/train.json', 'pipeline': [{'type': 'LoadImageFromFile', 'file_client_args': {'backend': 'disk'}}, {'type': 'LoadAnnotations', 'with_bbox': True, 'with_mask': True}, {'type': 'Resize', 'img_scale': [(1024, 1024)], 'keep_ratio': True, 'ratio_range': [0.8, 1.25], 'multiscale_mode': 'range'}, {'type': 'RandomCrop', 'crop_size': [1024, 1024], 'crop_type': 'absolute_range', 'recompute_bbox': True, 'allow_negative_crop': True}, {'type': 'FilterAnnotations', 'min_gt_bbox_wh': [0.01, 0.01]}, {'type': 'RandomFlip', 'flip_ratio': 0.5}, {'size': [1024, 1024], 'type': 'Pad'}], 'data_root': '/home/nils/datasets/cars/', 'img_prefix': 'raw'}}, {'std': [58.395, 57.12, 57.375], 'mean': [123.675, 116.28, 103.53], 'type': 'Normalize', 'to_rgb': True}, {'type': 'DefaultFormatBundle'}, {'keys': ['img', 'gt_bboxes', 'gt_labels'], 'type': 'Collect'}]}, 'samples_per_gpu': 4, 'workers_per_gpu': 2}
dims = [32, 64, 160, 256]
hooks = [{'type': 'TextLoggerHook', 'by_epoch': True}, {'type': 'MMDetWandbHook', 'by_epoch': True, 'interval': 10, 'init_kwargs': {'tags': ['backbone:VAN-B0', 'neck:FPN', 'head:FCOS', 'pretrained', 'schedule:1x', 'finetune-copy-paste'], 'entity': 'nkoch-aitastic', 'project': 'van-detection'}, 'log_checkpoint': True, 'num_eval_images': 100, 'log_checkpoint_metadata': True}]
model = {'neck': {'type': 'FPN', 'num_outs': 5, 'in_channels': [32, 64, 160, 256], 'start_level': 1, 'out_channels': 256, 'add_extra_convs': 'on_output', 'relu_before_extra_convs': True}, 'type': 'FCOS', 'backbone': {'type': 'VAN_Official', 'depths': [3, 3, 5, 2], 'init_cfg': {'type': 'Pretrained', 'checkpoint': 'models/van_tiny_754.pth.tar'}, 'norm_cfg': {'type': 'SyncBN', 'requires_grad': False}, 'act_layer': 'GELU', 'drop_rate': 0, 'embed_dims': [32, 64, 160, 256], 'drop_path_rate': 0.1}, 'test_cfg': {'nms': {'type': 'nms', 'iou_threshold': 0.5}, 'nms_pre': 1000, 'score_thr': 0.05, 'max_per_img': 100, 'min_bbox_size': 0}, 'bbox_head': {'type': 'FCOSHead', 'strides': [8, 16, 32, 64, 128], 'loss_cls': {'type': 'FocalLoss', 'alpha': 0.25, 'gamma': 2, 'loss_weight': 1, 'use_sigmoid': True}, 'loss_bbox': {'type': 'IoULoss', 'loss_weight': 1}, 'in_channels': 256, 'num_classes': 80, 'feat_channels': 256, 'stacked_convs': 4, 'loss_centerness': {'type': 'CrossEntropyLoss', 'loss_weight': 1, 'use_sigmoid': True}}, 'train_cfg': {'debug': False, 'assigner': {'type': 'MaxIoUAssigner', 'min_pos_iou': 0, 'neg_iou_thr': 0.4, 'pos_iou_thr': 0.5, 'ignore_iof_thr': -1}, 'pos_weight': -1, 'allowed_border': -1}}
runner = {'type': 'EpochBasedRunner', 'max_epochs': 12}
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'Audi_RS_6_Avant']
gpu_ids = [0, 1, 2]
norm_cfg = {'type': 'SyncBN', 'requires_grad': False}
work_dir = './work_dirs/fcos_van_b0_fpn_1x_coco_fcp_adam'
workflow = [['train', 1], ['val', 1]]
data_root = 'datasets/coco/'
load_from = None
log_level = 'INFO'
lr_config = {'power': 0.9, 'min_lr': 0, 'policy': 'poly', 'by_epoch': False}
optimizer = {'lr': 0.0001, 'type': 'AdamW', 'weight_decay': 0.0001}
evaluation = {'metric': ['bbox'], 'interval': 1}
image_size = [1024, 1024]
log_config = {'hooks': [{'type': 'TextLoggerHook', 'by_epoch': True}, {'type': 'MMDetWandbHook', 'by_epoch': True, 'interval': 10, 'init_kwargs': {'tags': ['backbone:VAN-B0', 'neck:FPN', 'head:FCOS', 'pretrained', 'schedule:1x', 'finetune-copy-paste'], 'entity': 'nkoch-aitastic', 'project': 'van-detection'}, 'log_checkpoint': True, 'num_eval_images': 100, 'log_checkpoint_metadata': True}], 'interval': 50}
pretrained = {'type': 'Pretrained', 'checkpoint': 'models/van_tiny_754.pth.tar'}
auto_resume = False
dist_params = {'backend': 'nccl'}
resume_from = None
custom_hooks = [{'type': 'NumClassCheckHook'}]
dataset_type = 'CocoDataset'
img_norm_cfg = {'std': [58.395, 57.12, 57.375], 'mean': [123.675, 116.28, 103.53], 'to_rgb': True}
load_pipeline = [{'type': 'LoadImageFromFile', 'file_client_args': {'backend': 'disk'}}, {'type': 'LoadAnnotations', 'with_bbox': True, 'with_mask': True}, {'type': 'Resize', 'img_scale': [(1024, 1024)], 'keep_ratio': True, 'ratio_range': [0.8, 1.25], 'multiscale_mode': 'range'}, {'type': 'RandomCrop', 'crop_size': [1024, 1024], 'crop_type': 'absolute_range', 'recompute_bbox': True, 'allow_negative_crop': True}, {'type': 'FilterAnnotations', 'min_gt_bbox_wh': [0.01, 0.01]}, {'type': 'RandomFlip', 'flip_ratio': 0.5}, {'size': [1024, 1024], 'type': 'Pad'}]
test_pipeline = [{'type': 'LoadImageFromFile'}, {'flip': False, 'type': 'MultiScaleFlipAug', 'img_scale': [(1333, 800)], 'transforms': [{'type': 'Resize', 'keep_ratio': True}, {'type': 'RandomFlip'}, {'std': [58.395, 57.12, 57.375], 'mean': [123.675, 116.28, 103.53], 'type': 'Normalize', 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'keys': ['img'], 'type': 'ImageToTensor'}, {'keys': ['img'], 'type': 'Collect'}]}]
custom_imports = {'imports': ['finetunecopypaste'], 'allow_failed_imports': False}
train_pipeline = [{'type': 'FineTuneCopyPaste', 'max_num_pasted': 100, 'supl_dataset_cfg': {'classes': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'Audi_RS_6_Avant'], 'ann_file': '/home/nils/datasets/cars/coco/train.json', 'pipeline': [{'type': 'LoadImageFromFile', 'file_client_args': {'backend': 'disk'}}, {'type': 'LoadAnnotations', 'with_bbox': True, 'with_mask': True}, {'type': 'Resize', 'img_scale': [(1024, 1024)], 'keep_ratio': True, 'ratio_range': [0.8, 1.25], 'multiscale_mode': 'range'}, {'type': 'RandomCrop', 'crop_size': [1024, 1024], 'crop_type': 'absolute_range', 'recompute_bbox': True, 'allow_negative_crop': True}, {'type': 'FilterAnnotations', 'min_gt_bbox_wh': [0.01, 0.01]}, {'type': 'RandomFlip', 'flip_ratio': 0.5}, {'size': [1024, 1024], 'type': 'Pad'}], 'data_root': '/home/nils/datasets/cars/', 'img_prefix': 'raw'}}, {'std': [58.395, 57.12, 57.375], 'mean': [123.675, 116.28, 103.53], 'type': 'Normalize', 'to_rgb': True}, {'type': 'DefaultFormatBundle'}, {'keys': ['img', 'gt_bboxes', 'gt_labels'], 'type': 'Collect'}]
cudnn_benchmark = True
file_client_args = {'backend': 'disk'}
optimizer_config = {}
checkpoint_config = {'by_epoch': True, 'max_keep_ckpts': 3}
