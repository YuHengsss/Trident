import os
import argparse
import trident
import custom_datasets
from myutils import append_experiment_result

from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(
        description='Trident evaluation with MMSeg')
    parser.add_argument('--config', default='./configs/cfg_voc20.py')
    parser.add_argument('--work-dir', default='./workdirs/')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show_dir',
        default='./show_dir/',
        help='directory to save visualizaion images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--sam_refine', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir
    cfg.model.sam_refinement = args.sam_refine

    runner = Runner.from_cfg(cfg)
    results = runner.test()

    results.update({'Model': cfg.model.model_type,
                    'CLIP': cfg.model.clip_type,
                    'VFM': cfg.model.vfm_model,
                    'Dataset': cfg.dataset_type})

    if runner.rank == 0:
        append_experiment_result('results.xlsx', [results])

    if runner.rank == 0:
        with open(os.path.join(cfg.work_dir, 'results.txt'), 'a') as f:
            f.write(os.path.basename(args.config).split('.')[0] + '\n')
            for k, v in results.items():
                f.write(k + ': ' + str(v) + '\n')

if __name__ == '__main__':
    main()