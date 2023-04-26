_base_ = [
    'fcos_van_b2_fpn_1x_coco_adam_newvan.py',
]


dims = [64, 128, 320, 512]
pretrained = dict(type='Pretrained', checkpoint='/home/nils/VAN-Detection/models/van-base_8xb128_in1k_20220501-conv.pth')
norm_cfg = dict(type='SyncBN', requires_grad=not bool(pretrained))  # train BN only when not using pretrained model


model = dict(
    backbone=dict(frozen_layers=[0,1,2]),


log_config = dict(
    hooks=[
        dict(type='MMDetWandbHook', 
          by_epoch=True, 
          init_kwargs=dict(
            entity="nkoch-aitastic",
            project='van-detection', 
            tags=[
              'backbone:VAN-B2',
              'backbone:frozen',
              'neck:FPN',
              'head:FCOS', 
              'pretrained',
              'schedule:1x',
              ]       
          ),
          interval=10,
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        dict(type='TextLoggerHook', by_epoch=True),
    ])

