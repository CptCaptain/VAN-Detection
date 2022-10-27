_base_ = [
    'fcos_van_b0_fpn_coco',
]

dims = [64, 128, 320, 512]

# model settings
model = dict(
    backbone=dict(
        embed_dims=dims,
        depths=[3, 3, 12, 3],
        init_cfg=dict(type='Pretrained', checkpoint='/content/models/van_small_811.pth.tar'),
        drop_path_rate=0.2
        ),
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
          num_eval_images=10,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
    ])

