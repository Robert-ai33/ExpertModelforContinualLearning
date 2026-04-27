"""
Directed Expert-based Class-Incremental Continual Learning

Supported datasets: wikics, ogbn-arxiv
Supported models: expert (DirectedExpertCL), naive_gcn (NaiveGCNCL)

Usage:
  python main.py --dataset wikics
  python main.py --dataset wikics --model naive_gcn --config_path ./configs/config_naive_gcn.yaml
  python main.py --dataset ogbn-arxiv --gpu 0
"""

import os
import yaml
import argparse
import torch
import numpy as np

from data import GraphDataset, TaskLoader
from models import DirectedExpertCL, NaiveGCNCL
from utils import seed_everything, CLMetric


# Default experiment settings per dataset
exp_settings = {
    'wikics': {
        'class_splits': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        'train_shots': 200,
        'valid_shots': 50,
        'test_shots': 100,
        'data_path': '../wiki-cs-dataset-master/dataset/',
    },
    'ogbn-arxiv': {
        # 40 classes, split into 10 sessions (4 classes each)
        'class_splits': [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]],
        'train_shots': 500,
        'valid_shots': 100,
        'test_shots': 200,
        'data_path': './data_files/',
    },
}


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(args, config):
    """Run a single experiment."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Get dataset-specific settings
    dataset_settings = exp_settings[args.dataset]

    # Override with command-line arguments if provided
    if args.class_splits:
        class_splits = eval(args.class_splits)
    else:
        class_splits = dataset_settings['class_splits']

    train_shots = args.train_shots if args.train_shots else dataset_settings['train_shots']
    valid_shots = args.valid_shots if args.valid_shots else dataset_settings['valid_shots']
    test_shots = args.test_shots if args.test_shots else dataset_settings['test_shots']
    data_path = args.data_path if args.data_path else dataset_settings['data_path']

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Class Splits: {class_splits}")
    print(f"Train/Valid/Test shots: {train_shots}/{valid_shots}/{test_shots}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = GraphDataset(dataset=args.dataset, data_path=data_path)

    # Create task loader
    task_loader = TaskLoader(
        batch_size=config['batch_size'],
        dataset=dataset,
        class_splits=class_splits,
        train_shots=train_shots,
        valid_shots=valid_shots,
        test_shots=test_shots,
    )

    # Run trials
    results = {'avg_acc': [], 'avg_fgt': [], 'last_acc': []}

    for trial in range(args.ntrials):
        seed = config['seed'][trial] if trial < len(config['seed']) else trial
        seed_everything(seed)

        print(f"\n--- Trial {trial + 1}/{args.ntrials} (seed={seed}) ---")

        result_logger = CLMetric()

        if args.model == 'naive_gcn':
            model = NaiveGCNCL(
                task_loader=task_loader,
                result_logger=result_logger,
                config=config,
                checkpoint_path=args.ckpt_path,
                seed=seed,
                device=device,
            )
        else:
            model = DirectedExpertCL(
                task_loader=task_loader,
                result_logger=result_logger,
                config=config,
                checkpoint_path=args.ckpt_path,
                seed=seed,
                device=device,
            )

        result_logger = model.fit(trial)
        avg_acc, avg_fgt, _, last_acc = result_logger.get_results()

        results['avg_acc'].append(avg_acc)
        results['avg_fgt'].append(avg_fgt)
        results['last_acc'].append(last_acc)

        print(f"Trial {trial + 1}: Avg ACC={avg_acc:.4f}, "
              f"Avg FGT={avg_fgt:.4f}, Last ACC={last_acc:.4f}")

    # Print final results
    print(f"\n{'='*60}")
    print(f"[{args.dataset}] Final Results ({args.ntrials} trials):")
    print(f"  Avg ACC: {np.mean(results['avg_acc']):.4f} "
          f"+/- {np.std(results['avg_acc']):.4f}")
    print(f"  Avg FGT: {np.mean(results['avg_fgt']):.4f} "
          f"+/- {np.std(results['avg_fgt']):.4f}")
    print(f"  Last ACC: {np.mean(results['last_acc']):.4f} "
          f"+/- {np.std(results['last_acc']):.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Directed Expert CL on Graphs'
    )

    # Dataset
    parser.add_argument('--dataset', type=str, default='wikics',
                        choices=['wikics', 'ogbn-arxiv'],
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset files (default: auto per dataset)')

    # Class splits and data split sizes
    parser.add_argument('--class_splits', type=str, default=None,
                        help='Custom class splits, e.g., "[[0,1],[2,3],[4,5]]"')
    parser.add_argument('--train_shots', type=int, default=None,
                        help='Training samples per class')
    parser.add_argument('--valid_shots', type=int, default=None,
                        help='Validation samples per class')
    parser.add_argument('--test_shots', type=int, default=None,
                        help='Test samples per class')

    # Model
    parser.add_argument('--model', type=str, default='expert',
                        choices=['expert', 'naive_gcn'],
                        help='Model type: expert (DirectedExpertCL) or naive_gcn (NaiveGCNCL)')
    parser.add_argument('--config_path', type=str,
                        default='./configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/',
                        help='Path to save checkpoints')

    # Training
    parser.add_argument('--ntrials', type=int, default=1,
                        help='Number of experiment trials')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config_path)
    if 'default' in config:
        config = config['default']

    # Create directories
    os.makedirs(args.ckpt_path, exist_ok=True)

    # Run experiment
    run_experiment(args, config)
