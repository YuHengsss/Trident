_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_context60.txt',
    prob_thd=0.15,  # 0.15 OURS   0.05 CLIP
    coarse_thresh = 0.15,
)

# dataset settings
dataset_type = 'PascalContext60Dataset'
data_root = '/home/yuheng/datasets/VOCdevkit/VOC2010'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 576), keep_ratio=True),
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
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))