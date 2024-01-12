# 说明
## 统一使用图像生成keypoint
## 统一使用关键点检测的score
## 只使用收集的图像，去除NTU Huhman36M
model = dict(
    type='SkeletonGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        edge_importance_weighting=True,
        graph_cfg=dict(layout='SportBody', strategy='spatial')),
    cls_head=dict(
        type='STGCNHead',
        #num_classes=60,
        num_classes=3,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'PoseDataset'
ann_file_train = '/home/Project/TsportSkeleton/train-v3.pkl'
ann_file_val = '/home/Project/TsportSkeleton/val-v3.pkl'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=1),
    dict(type='PoseDecode'),
    dict(type='PosePerspectiveTranslateAugCJH', x_factor=0.1, y_factor=0.3),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalizeCJH'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=1),
    dict(type='PoseDecode'),
    dict(type='PosePerspectiveTranslateAugCJH', x_factor=0.1, y_factor=0.3),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalizeCJH'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM'),
    dict(type='PoseNormalizeCJH'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=1024,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 50])
total_epochs = 300
checkpoint_config = dict(interval=4)
evaluation = dict(interval=4, metrics=['top_k_accuracy'])
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/sport_stgcn_v3/'
load_from = None
resume_from = None
#workflow = [('val', 1), ('train', 1)] # 代表着一个epoch的行为
workflow = [('train', 1)]
