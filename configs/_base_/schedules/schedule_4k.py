
gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(
  type='AdamW', 
  lr=0.0001*gpu_multiples, 
  weight_decay=0.0001,
  )
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
interval = 4000
runner = dict(type='IterBasedRunner', max_iters=10*interval)
checkpoint_config = dict(by_epoch=False, interval=interval, max_keep_ckpts=3)
evaluation = dict(interval=interval, metric=['bbox'])
# evaluation = dict(interval=50, metric=['bbox', 'proposal'])