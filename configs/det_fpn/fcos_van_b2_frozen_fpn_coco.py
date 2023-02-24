_base_ = [
    'fcos_van_b2_fpn_coco.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=False)
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='/home/nils/VAN-Detection/models/van_base_828.pth.tar'),
        frozen=True,
        norm_cfg=norm_cfg,
      ),
    )


log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False, reset_flag=True),
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
              'frozen-backbone'
              ]       
          ),
          interval=10,
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
    ])

  

