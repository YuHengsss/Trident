_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_voc21.txt',
    prob_thd= 0.2,     # 0.2  OURS  0.1 CLIP-B/16
    coarse_thresh = 0.2,
)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/home/yuheng/datasets/VOCdevkit/VOC2012'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))