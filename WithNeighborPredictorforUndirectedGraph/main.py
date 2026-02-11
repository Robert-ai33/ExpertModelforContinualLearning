"""
Expert-based Class-Incremental Continual Learning Framework

框架特点:
1. 实验场景:
   - 每个阶段有一个类别范围
   - 子图包含类别范围内节点及其直接邻居
   - 类别内边全部保留，类别内到类别外边保留，类别外之间边删除

2. 模型结构:
   - 每个阶段一个专家（分类器 + 邻居预测器）
   - 分类器: 单层GCN + 分类头
   - 邻居预测器: 直接使用原始特征

3. 训练策略:
   - 前cls_epochs: 训练分类器
   - 后epochs-cls_epochs: 训练邻居预测器

4. 测试策略:
   - 用邻居预测准确度选择专家
   - 使用选中专家的分类器进行预测
"""

import os
import yaml
import argparse
import torch
import numpy as np

from data import TextDataset, TaskLoader
from models import ExpertCL
from utils import seed_everything, CLMetric


# Experiment settings - directly specify class splits
exp_settings = {
    'cora': {
        # Default: standard order
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
        # 10 classes total, split into 5 sessions
        'class_splits': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        'train_shots': 200,
        'valid_shots': 50,
        'test_shots': 50,
    },
}


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(args, config):
    """Run a single experiment."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Get dataset settings
    dataset_settings = exp_settings[args.dataset]
    
    # Use custom class_splits if provided, otherwise use default
    if args.class_splits:
        class_splits = eval(args.class_splits)  # Parse string to list
    else:
        class_splits = dataset_settings['class_splits']
    
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Class Splits: {class_splits}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load dataset
    text_dataset = TextDataset(dataset=args.dataset, data_path=args.data_path)
    
    # Create task loader
    task_loader = TaskLoader(
        batch_size=config['batch_size'],
        text_dataset=text_dataset,
        class_splits=class_splits,
        train_shots=dataset_settings['train_shots'],
        valid_shots=dataset_settings['valid_shots'],
        test_shots=dataset_settings['test_shots'],
    )
    
    # Run multiple trials
    results = {'avg_acc': [], 'avg_fgt': [], 'last_acc': []}
    
    for trial in range(args.ntrials):
        seed = config['seed'][trial] if trial < len(config['seed']) else trial
        seed_everything(seed)
        
        print(f"\n--- Trial {trial + 1}/{args.ntrials} (seed={seed}) ---")
        
        # Create result logger
        result_logger = CLMetric()
        
        # Create model
        model = ExpertCL(
            task_loader=task_loader,
            result_logger=result_logger,
            config=config,
            checkpoint_path=args.ckpt_path,
            dataset=args.dataset,
            seed=seed,
            device=device,
        )
        
        # Train and evaluate
        result_logger = model.fit(trial)
        avg_acc, avg_fgt, _, last_acc = result_logger.get_results()
        
        results['avg_acc'].append(avg_acc)
        results['avg_fgt'].append(avg_fgt)
        results['last_acc'].append(last_acc)
        
        print(f"Trial {trial + 1}: Avg ACC={avg_acc:.4f}, Avg FGT={avg_fgt:.4f}, Last ACC={last_acc:.4f}")
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"Final Results ({args.ntrials} trials):")
    print(f"  Avg ACC: {np.mean(results['avg_acc']):.4f} ± {np.std(results['avg_acc']):.4f}")
    print(f"  Avg FGT: {np.mean(results['avg_fgt']):.4f} ± {np.std(results['avg_fgt']):.4f}")
    print(f"  Last ACC: {np.mean(results['last_acc']):.4f} ± {np.std(results['last_acc']):.4f}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expert-based Class-Incremental Continual Learning')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cora', 
                        choices=['cora', 'citeseer', 'wikics'],
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data_files/',
                        help='Path to dataset files')
    
    # Class splits - can directly specify custom splits
    parser.add_argument('--class_splits', type=str, default=None,
                        help='Custom class splits, e.g., "[[0,1],[2,3],[4,5]]"')
    
    # Model
    parser.add_argument('--config_path', type=str, default='./configs/ExpertCL.yaml',
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
    os.makedirs(args.data_path, exist_ok=True)
    
    # Run experiment
    run_experiment(args, config)
