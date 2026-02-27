"""
Spectral Clustering Expert-based Class-Incremental Continual Learning

Framework:
1. Each session has a set of target classes
2. Each session trains one expert with 3 MLPs:
   - Neighbor Predictor (for expert selection)
   - Cross-class Neighbor Predictor (for edge removal)
   - Same-class Predictor (for edge addition)
3. Inference: modify graph -> regularized spectral clustering -> purity
4. Evaluation: cluster purity on joint test set

Usage:
    python main.py --dataset cora
    python main.py --dataset ogbn-arxiv
    python main.py --dataset wikics --class_splits "[[0,1],[2,3],[4,5],[6,7],[8,9]]"
"""

import os
import yaml
import argparse
import torch
import numpy as np

from data import GraphDataset, TaskLoader
from models import SpectralExpertCL
from utils import seed_everything


# Default experiment settings per dataset
exp_settings = {
    'cora': {
        'class_splits': [[0, 1], [2, 3], [4, 5]],
        'train_shots': 100,
        'valid_shots': 50,
        'test_shots': 100,
    },
    'citeseer': {
        'class_splits': [[0, 1], [2, 3], [4, 5]],
        'train_shots': 100,
        'valid_shots': 50,
        'test_shots': 100,
    },
    'wikics': {
        'class_splits': [[1, 2], [3, 4], [5, 6], [7, 8, 9]],
        'train_shots': 200,
        'valid_shots': 50,
        'test_shots': 50,
    },
    'coauthor-cs': {
        'class_splits': [[0, 1],[2, 3], [4, 5], [6, 7, 8], [9, 10, 11],[12, 13, 14]],
        'train_shots': 200,
        'valid_shots': 50,
        'test_shots': 100,
    },
    'amazon-computers': {
        'class_splits': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        'train_shots': 200,
        'valid_shots': 50,
        'test_shots': 100,
    },
    'ogbn-arxiv': {
        'class_splits': [[8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
        'train_shots': 500,
        'valid_shots': 100,
        'test_shots': 200,
    },
    'ogbn-products': {
        'class_splits': [[12,13,14,15],[16,17,18,19],[20,21,22,23],[24,25,26,27]],
        'train_shots': 500,
        'valid_shots': 100,
        'test_shots': 200,
    },
}


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(args, config):
    """Run a single experiment (possibly multiple trials)."""
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    # Get dataset-specific settings
    if args.dataset in exp_settings:
        dataset_settings = exp_settings[args.dataset]
    else:
        # Unknown dataset: require class_splits
        if not args.class_splits:
            raise ValueError(
                f"Unknown dataset '{args.dataset}'. "
                f"Please specify --class_splits."
            )
        dataset_settings = {
            'class_splits': eval(args.class_splits),
            'train_shots': args.train_shots,
            'valid_shots': args.valid_shots,
            'test_shots': args.test_shots,
        }

    # Override with command-line arguments if provided
    if args.class_splits:
        class_splits = eval(args.class_splits)
    else:
        class_splits = dataset_settings['class_splits']

    train_shots = args.train_shots if args.train_shots else dataset_settings['train_shots']
    valid_shots = args.valid_shots if args.valid_shots else dataset_settings['valid_shots']
    test_shots = args.test_shots if args.test_shots else dataset_settings['test_shots']

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Class Splits: {class_splits}")
    print(f"Train/Valid/Test shots: {train_shots}/{valid_shots}/{test_shots}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load dataset
    graph_dataset = GraphDataset(
        dataset=args.dataset, data_path=args.data_path
    )

    # Create task loader
    task_loader = TaskLoader(
        batch_size=config['batch_size'],
        graph_dataset=graph_dataset,
        class_splits=class_splits,
        train_shots=train_shots,
        valid_shots=valid_shots,
        test_shots=test_shots,
    )

    # Run multiple trials
    results = {'avg_purity': [], 'last_purity': []}

    for trial_idx in range(args.ntrials):
        seed = config['seed'][trial_idx] if trial_idx < len(config['seed']) else trial_idx
        seed_everything(seed)

        print(f"\n--- Trial {trial_idx + 1}/{args.ntrials} (seed={seed}) ---")

        # Create model
        model = SpectralExpertCL(
            task_loader=task_loader,
            config=config,
            checkpoint_path=args.ckpt_path,
            dataset=args.dataset,
            seed=seed,
            device=device,
        )

        # Train and evaluate
        purity_metric = model.fit(trial_idx)
        avg_purity, last_purity = purity_metric.get_summary()

        results['avg_purity'].append(avg_purity)
        results['last_purity'].append(last_purity)

        print(f"\nTrial {trial_idx + 1}: "
              f"Avg Purity={avg_purity:.4f}, "
              f"Last Purity={last_purity:.4f}")

    # Print final results
    print(f"\n{'='*60}")
    print(f"Final Results ({args.ntrials} trials):")
    print(f"  Avg Purity: "
          f"{np.mean(results['avg_purity']):.4f} ± "
          f"{np.std(results['avg_purity']):.4f}")
    print(f"  Last Purity: "
          f"{np.mean(results['last_purity']):.4f} ± "
          f"{np.std(results['last_purity']):.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Spectral Clustering Expert-based Continual Learning'
    )

    # Dataset
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Dataset name (cora, citeseer, wikics, '
                             'ogbn-arxiv, ogbn-products, etc.)')
    parser.add_argument('--data_path', type=str, default='./data_files/',
                        help='Path to dataset files')

    # Class splits and data split sizes
    parser.add_argument('--class_splits', type=str, default=None,
                        help='Custom class splits, e.g., "[[0,1],[2,3],[4,5]]"')
    parser.add_argument('--train_shots', type=int, default=None,
                        help='Training samples per class')
    parser.add_argument('--valid_shots', type=int, default=None,
                        help='Validation samples per class')
    parser.add_argument('--test_shots', type=int, default=None,
                        help='Test samples per class')

    # Model config
    parser.add_argument('--config_path', type=str,
                        default='./configs/config.yaml',
                        help='Path to config YAML file')
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
    os.makedirs(args.data_path, exist_ok=True)

    # Run
    run_experiment(args, config)
