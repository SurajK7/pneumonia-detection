_base_ = [
    'retinanet_r50_fpn.py',
    'rsna_pneumonia_detection.py',
    'schedule_rsna.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
