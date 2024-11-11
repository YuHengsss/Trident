# base configurations
sam_ckpt =  '/home/yuheng/project/trident-pure/exclude/sam_vit_b_01ec64.pth'
sam_refinement = False

model = dict(
    type='Trident',
    clip_type= 'openai',
    model_type= 'ViT-B/16',
    vfm_model='dino',
    checkpoint=None,
    beta=1.2,
    cos_fac = 1.2,
    refine_neg_cos = True,
    slide_stride=224,
    slide_crop=336,
    sam_model_type='vit_b',
    sam_iou_thresh=0.80,
    coarse_thresh=0.2,
    minimal_area=225,
    sam_ckpt=sam_ckpt,
    sam_refinement=sam_refinement,
)
# ('openai', 'ViT-B/16')
# ('laion2b_s32b_b79k', 'ViT-H-14')

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, alpha=1.0, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=5))