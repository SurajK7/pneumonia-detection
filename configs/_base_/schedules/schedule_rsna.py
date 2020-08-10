# optimizer
#lr = 0.01 for bs=8
optimizer = dict(type='SGD', lr=2.5e-3, momentum=0.9, weight_decay=0.0001, nesterov=True)#bs=4
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-3,
    step=[3, 6, 9])
total_epochs = 10
