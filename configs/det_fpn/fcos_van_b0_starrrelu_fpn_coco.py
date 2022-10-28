_base_ = [
    'fcos_van_b0_fpn_coco.py',
]

model = dict(
    type='FCOS',
    backbone=dict(
        # This is new, taken from MetaFromer baselines for Vision
        act_layer='StarReLU',
        init_cfg=dict(type='Pretrained', checkpoint='/content/models/van_tiny_754.pth.tar'),
      ),
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
              'StarReLU',
              ]       
          ),
          interval=10,
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
    ])

  
