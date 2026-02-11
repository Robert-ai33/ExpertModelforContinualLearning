"""
Directed Expert-based Class-Incremental Continual Learning on WikiCS

Features:
1. Directed and undirected graph support
2. Classification expert: GCN + classifier on undirected graph
3. Out-neighbor predictor: predicts A->B on directed graph
4. In-neighbor predictor: predicts A<-B on directed graph
5. Expert selection based on directed neighbor prediction accuracy
"""

import os
import yaml
import argparse
import torch
import numpy as np

from data import WikiCSDataset, TaskLoader
from models import DirectedExpertCL
from utils import seed_everything, CLMetric


# WikiCS: 10 classes, default split into 5 sessions
exp_settings = {
    'class_splits': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
    'train_shots': 200,
    'valid_shots': 50,
    'test_shots': 50,
}


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(args, config):
    """Run a single experiment."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Use custom class_splits if provided
    if args.class_splits:
        class_splits = eval(args.class_splits)
    else:
        class_splits = exp_settings['class_splits']

    print(f"\n{'='*60}")
    print(f"Directed Expert CL on WikiCS")
    print(f"Class Splits: {class_splits}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = WikiCSDataset(data_path=args.data_path)

    # Create task loader
    task_loader = TaskLoader(
        batch_size=config['batch_size'],
        dataset=dataset,
        class_splits=class_splits,
        train_shots=exp_settings['train_shots'],
        valid_shots=exp_settings['valid_shots'],
        test_shots=exp_settings['test_shots'],
    )

    # Run trials
    results = {'avg_acc': [], 'avg_fgt': [], 'last_acc': []}

    for trial in range(args.ntrials):
        seed = config['seed'][trial] if trial < len(config['seed']) else trial
        seed_everything(seed)

        print(f"\n--- Trial {trial + 1}/{args.ntrials} (seed={seed}) ---")

        result_logger = CLMetric()

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
    print(f"Final Results ({args.ntrials} trials):")
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
        description='Directed Expert CL on WikiCS'
    )

    # Dataset
    parser.add_argument('--data_path', type=str,
                        default='../wiki-cs-dataset-master/dataset/',
                        help='Path to WikiCS dataset directory (containing data.json)')

    # Class splits
    parser.add_argument('--class_splits', type=str, default=None,
                        help='Custom class splits, e.g., "[[0,1],[2,3],[4,5]]"')

    # Model
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
