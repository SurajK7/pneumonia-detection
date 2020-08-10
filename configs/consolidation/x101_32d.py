_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'))
load_from='https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth'
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001, nesterov=True)#bs=4
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=6)
log_config = dict(interval=200)