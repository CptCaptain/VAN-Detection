# optimizer
optimizer = dict(
  type='AdamW', 
  lr=0.0001, 
  weight_decay=0.0001,
  )
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1, metric=['bbox'])
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=3)