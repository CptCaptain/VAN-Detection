_base_ = [
    'fcos_van_b0_fpn_dcn_1x_coco_adam_scp_bof.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=False)
model = dict(
    backbone=dict(
        frozen_stages=3,
      ),
    )


log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True, reset_flag=True),
        dict(type='MMDetWandbHook', 
          by_epoch=True, 
          init_kwargs=dict(
            entity="nkoch-aitastic",
            project='van-detection', 
            tags=[
              'backbone:VAN-B0', 
              'neck:FPN',
              'head:FCOS', 
              'schedule:1x-adam',
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

  
workflow=[('train', 12),]

