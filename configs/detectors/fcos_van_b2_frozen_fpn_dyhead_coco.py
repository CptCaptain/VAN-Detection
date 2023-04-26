_base_ = [
    'fcos_van_b2_fpn_coco.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=False)
dims = [64, 128, 320, 512]
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='/content/models/van_small_811.pth.tar'),
        frozen=True,
        norm_cfg=norm_cfg,
      ),
    neck=[
        dict(
            type='FPN',
            in_channels=dims,
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)
      ]
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
              'neck:FPN+DyHead',
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

  

