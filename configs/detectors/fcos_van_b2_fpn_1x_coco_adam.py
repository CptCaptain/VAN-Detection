_base_ = [
    'fcos_van_b0_fpn_1x_coco_adam.py',
]

dims = [64, 128, 320, 512]
pretrained = dict(type='Pretrained', checkpoint='models/van-base_8xb128_in1k_20220501-6a4cc31b.pth')

# model settings
model = dict(
    backbone=dict(type='VAN', _delete=True, arch='b2', drop_path_rate=0.1),
    neck=dict(in_channels=dims),
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
              'backbone:VAN-B2', 
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


data = dict(samples_per_gpu=1)
