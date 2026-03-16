"""
CommunityExpertCL - Main entry point.

Usage:
  python main.py --dataset cora --gpu 0
  python main.py --dataset cora --model bare --config_path ./configs/config_bare.yaml
  python main.py --dataset cora --model ewc  --config_path ./configs/config_ewc.yaml
  python main.py --dataset cora --model mas  --config_path ./configs/config_mas.yaml
  python main.py --dataset cora --model gem  --config_path ./configs/config_gem.yaml
  python main.py --dataset cora --model twp  --config_path ./configs/config_twp.yaml
  python main.py --dataset cora --model lwf  --config_path ./configs/config_lwf.yaml
  python main.py --dataset cora --model joint --config_path ./configs/config_joint.yaml
  python main.py --dataset coauthor-cs --gpu 0 --amp
"""

import os
import argparse
import yaml
import numpy as np

import torch

from data import GraphDataset, TaskLoader
from models import CommunityExpertCL
from models.baselines import BASELINE_MODELS
from utils import seed_everything


# Dataset-specific experiment settings
EXP_SETTINGS = {
    'cora': {
        'class_splits': [[0, 1], [2, 3], [4, 5, 6]],
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'citeseer': {
        'class_splits': [[0, 1], [2, 3], [4, 5]],
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'cora-full': {
        'class_splits': [
            [0,1,2,3,4,5,6], [7,8,9,10,11,12,13], [14,15,16,17,18,19,20],
            [21,22,23,24,25,26,27], [28,29,30,31,32,33,34], [35,36,37,38,39,40,41],
            [42,43,44,45,46,47,48], [49,50,51,52,53,54,55],[56,57,58,59,60,61,62],[63,64,65,66,67,68,69]
        ],
        'split_S': 5,
        'split_t': 2,
        'split_v': 1,
    },
    'coauthor-cs': {
        'class_splits': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13, 14]],
        'split_S': 5,
        'split_t': 1,
        'split_v': 1,
    },
    'amazon-computers': {
        'class_splits': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        'split_S': 5,
        'split_t': 1,
        'split_v': 1,
    },
    'wikics': {
        'class_splits': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        'split_S': 5,
        'split_t': 3,
        'split_v': 1,
    },
    'ogbn-arxiv': {
        'class_splits': [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19],[20,21,22,23],[24,25,26,27],[28,29,30,31],[32,33,34,35],[36,37,38,39]],
        'split_S': 10,
        'split_t': 3,
        'split_v': 2,
    },
    'ogbn-products': {
        'class_splits': [[6,7,8,9],[10,11,12,13],[14,15,16,17],[18,19,20,21],[22,23,24,25]],
        'split_S': 10,
        'split_t': 1,
        'split_v': 1,
    },
}


def main():
    parser = argparse.ArgumentParser(description='CommunityExpertCL')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=list(EXP_SETTINGS.keys()))
    parser.add_argument('--data_path', type=str, default='./data_files/')
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml')
    parser.add_argument('--ntrials', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='expert',
                        choices=['expert'] + list(BASELINE_MODELS.keys()),
                        help='Model to use (default: expert)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training (AMP)')
    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    config = full_config['default']

    # Merge dataset-specific settings
    exp = EXP_SETTINGS[args.dataset]
    config['class_splits'] = exp['class_splits']
    config['split_S'] = exp.get('split_S', config.get('split_S', 5))
    config['split_t'] = exp.get('split_t', config.get('split_t', 3))
    config['split_v'] = exp.get('split_v', config.get('split_v', 1))
    config['use_amp'] = args.amp

    # Device
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Class splits: {config['class_splits']}")
    print(f"Split ratio: t/S={config['split_t']}/{config['split_S']}, "
          f"v/S={config['split_v']}/{config['split_S']}")
    print(f"AMP: {'enabled' if args.amp else 'disabled'}")

    # Seeds
    seeds = config.get('seed', [0, 1, 2, 3, 4])
    ntrials = min(args.ntrials, len(seeds))

    # Run trials
    all_acc = []

    for trial in range(ntrials):
        seed = seeds[trial]
        print(f"\n{'#'*60}")
        print(f"Trial {trial + 1}/{ntrials}, Seed: {seed}")
        print(f"{'#'*60}")

        seed_everything(seed)

        graph_dataset = GraphDataset(args.dataset, args.data_path)

        task_loader = TaskLoader(
            batch_size=config.get('batch_size', 256),
            graph_dataset=graph_dataset,
            class_splits=config['class_splits'],
            split_S=config['split_S'],
            split_t=config['split_t'],
            split_v=config['split_v'],
        )

        if args.model == 'expert':
            model = CommunityExpertCL(
                task_loader=task_loader,
                config=config,
                device=device,
            )
        else:
            model_cls = BASELINE_MODELS[args.model]
            model = model_cls(
                task_loader=task_loader,
                config=config,
                device=device,
            )

        results = model.fit(trial)

        last_acc = results['joint_acc'][-1]
        all_acc.append(last_acc)

        print(f"\nTrial {trial + 1} Summary:")
        print(f"  Last Joint Accuracy: {last_acc:.4f}")

    # Final summary across trials
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY ({ntrials} trials)")
    print(f"{'='*60}")
    print(f"Joint Accuracy: {np.mean(all_acc):.4f} "
          f"\u00b1 {np.std(all_acc):.4f}")


if __name__ == '__main__':
    main()
