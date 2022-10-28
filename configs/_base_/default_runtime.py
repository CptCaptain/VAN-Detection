# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        dict(type='MMDetWandbHook', 
          by_epoch=False, # The Wandb logger is also supported, It requires `wandb` to be installed.
          init_kwargs=dict(
            entity="nkoch-aitastic", # The entity used to log on Wandb   
            project='van-detection',        
          ),
          interval=10,
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        # MMDetWandbHook is mmdet implementation of WandbLoggerHook. ClearMLLoggerHook, DvcliveLoggerHook, MlflowLoggerHook, NeptuneLoggerHook, PaviLoggerHook, SegmindLoggerHook are also supported based on MMCV implementation.
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
