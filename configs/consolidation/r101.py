_base_ = [
    '/home/suraj/mmdetection/configs/retinanet/retinanet_r101_fpn_1x_coco.py'
]

optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0001)
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8)
log_config = dict(interval=100)
# resume_from='/home/suraj/mmdetection/work_dirs/keras_retinanet/epoch_1.pth'