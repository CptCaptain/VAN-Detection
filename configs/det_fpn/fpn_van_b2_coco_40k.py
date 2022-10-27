_base_ = [
    '../_base_/models/fpn_van.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]
dims = [64, 128, 320, 512]

# commented out due to it not fitting into a 16GB GPU atm

# model settings
# model = dict(
#     type='RetinaNet',
#     backbone=dict(
#         embed_dims=dims,
#         depths=[3, 3, 12, 3],
#         init_cfg=dict(type='Pretrained', checkpoint='/content/models/van_small_811.pth.tar'),
#         drop_path_rate=0.2
#         ),
#     neck=dict(in_channels=dims),
#     bbox_head=dict(num_classes=80)
#     )


gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(
  type='AdamW', 
  lr=0.0001*gpu_multiples, 
  weight_decay=0.0001,
  # Freeze the backbone
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'backbone': dict(lr_mult=0, decay_mult=0),
    #     },
    # ),
  )
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
# evaluation = dict(interval=8000//gpu_multiples, metric=['bbox', 'proposal'])
evaluation = dict(interval=50, metric='mIoU')
data = dict(samples_per_gpu=4)
