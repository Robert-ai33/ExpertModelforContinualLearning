"""
Expert count ablation study for LiteExpertCL.

Usage:
  python run_expert_ablation.py --dataset cora --min_experts 2 --max_experts 5 --gpu 0
  python run_expert_ablation.py --dataset cora-full --min_experts 8 --max_experts 17 --gpu 0
  python run_expert_ablation.py --dataset coauthor-cs --min_experts 3 --max_experts 10 --step 2 --gpu 0

Outputs:
  results/expert_ablation/<dataset>/expert_ablation_micro.png     - Joint micro accuracy line plot
  results/expert_ablation/<dataset>/expert_ablation_macro.png     - Joint macro accuracy line plot
  results/expert_ablation/<dataset>/expert_ablation_heatmaps.png  - Accuracy matrix heatmaps grid
  results/expert_ablation/<dataset>/expert_ablation.json          - Raw results
  results/expert_ablation/<dataset>/legend.png                    - Legend for this dataset
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data import GraphDataset, TaskLoader
from models import LiteExpertCL
from utils import seed_everything
from main import EXP_SETTINGS

ABLATION_MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'p', 'h', '<', '>', 'd', 'H', '8']


def run_with_experts(n_experts, task_loader, config, device, ntrials, seeds):
    cfg = dict(config)
    cfg['max_experts'] = n_experts

    all_results = []
    for trial in range(ntrials):
        seed_everything(seeds[trial])
        model = LiteExpertCL(task_loader=task_loader, config=cfg, device=device)
        results = model.fit(trial)
        all_results.append(results)

    num_sessions = len(all_results[0]['joint_acc'])
    avg_joint_acc = [np.mean([r['joint_acc'][s] for r in all_results]) for s in range(num_sessions)]
    avg_joint_macro = [np.mean([r['joint_macro_acc'][s] for r in all_results]) for s in range(num_sessions)]

    avg_matrix = []
    for s in range(num_sessions):
        row_len = s + 1
        avg_row = [np.mean([r['acc_matrix'][s][t] for r in all_results]) for t in range(row_len)]
        avg_matrix.append(avg_row)

    return {
        'joint_acc': avg_joint_acc,
        'joint_macro_acc': avg_joint_macro,
        'acc_matrix': avg_matrix,
    }


def plot_lines(all_data, expert_counts, num_sessions, dataset, out_dir):
    sessions = list(range(num_sessions))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(expert_counts) - 1, 1)) for i in range(len(expert_counts))]

    for metric, fname in [
        ('joint_acc', 'expert_ablation_micro.png'),
        ('joint_macro_acc', 'expert_ablation_macro.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, n_exp in enumerate(expert_counts):
            mkr = ABLATION_MARKERS[i % len(ABLATION_MARKERS)]
            vals = all_data[n_exp][metric]
            ax.plot(sessions, vals, marker=mkr, markersize=8, color=colors[i],
                    label=f'{n_exp} experts', linewidth=1.8)

        ax.set_xlabel('Session', fontsize=13)
        ax.set_xticks(sessions)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Saved: {os.path.join(out_dir, fname)}")


def plot_ablation_legend(expert_counts, out_dir):
    """Generate legend image specific to this dataset's expert counts."""
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(expert_counts) - 1, 1)) for i in range(len(expert_counts))]

    fig, ax = plt.subplots(figsize=(12, 1))
    for i, n_exp in enumerate(expert_counts):
        mkr = ABLATION_MARKERS[i % len(ABLATION_MARKERS)]
        ax.plot([], [], marker=mkr, markersize=8, color=colors[i],
                label=f'{n_exp} experts', linewidth=1.8)
    ax.legend(fontsize=10, loc='center', ncol=len(expert_counts), frameon=False)
    ax.axis('off')
    fig.tight_layout()
    legend_path = os.path.join(out_dir, 'legend.png')
    fig.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved legend: {legend_path}")


def plot_heatmaps(all_data, expert_counts, num_sessions, dataset, out_dir):
    n = len(expert_counts)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, n_exp in enumerate(expert_counts):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        mat = all_data[n_exp]['acc_matrix']

        full_mat = np.full((num_sessions, num_sessions), np.nan)
        for s, row in enumerate(mat):
            for t, val in enumerate(row):
                full_mat[s][t] = val * 100

        masked = np.ma.array(full_mat, mask=np.isnan(full_mat))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')

        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect='equal', origin='upper')
        ax.set_title(f'{n_exp} experts', fontsize=13, fontweight='bold')
        ax.set_xlabel('Tasks', fontsize=10)
        ax.set_ylabel('Tasks', fontsize=10)

        if num_sessions <= 10:
            ax.set_xticks(range(num_sessions))
            ax.set_yticks(range(num_sessions))
            ax.set_xticklabels(range(num_sessions), fontsize=8)
            ax.set_yticklabels(range(num_sessions), fontsize=8)
        else:
            step = max(1, num_sessions // 6)
            ticks = list(range(0, num_sessions, step))
            if num_sessions - 1 not in ticks:
                ticks.append(num_sessions - 1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(ticks, fontsize=8)
            ax.set_yticklabels(ticks, fontsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis('off')

    fig.suptitle(f'{dataset} - Expert Ablation Heatmaps', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'expert_ablation_heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, 'expert_ablation_heatmaps.png')}")


def main():
    parser = argparse.ArgumentParser(description='Expert count ablation study')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=list(EXP_SETTINGS.keys()))
    parser.add_argument('--data_path', type=str, default='./data_files/')
    parser.add_argument('--min_experts', type=int, required=True,
                        help='Minimum number of experts')
    parser.add_argument('--max_experts', type=int, required=True,
                        help='Maximum number of experts')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size for expert count range')
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--svd_dim', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    exp = EXP_SETTINGS[args.dataset]

    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, 'configs', 'config_lite.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)['default']

    config['class_splits'] = exp['class_splits']
    config['split_S'] = exp.get('split_S', config.get('split_S', 5))
    config['split_t'] = exp.get('split_t', config.get('split_t', 3))
    config['split_v'] = exp.get('split_v', config.get('split_v', 1))
    config['use_amp'] = args.amp

    seeds = config.get('seed', [0, 1, 2, 3, 4])
    ntrials = min(args.ntrials, len(seeds))

    expert_counts = list(range(args.min_experts, args.max_experts + 1, args.step))
    print(f"Dataset: {args.dataset}")
    print(f"Expert counts to test: {expert_counts}")
    print(f"Trials per setting: {ntrials}")

    out_dir = os.path.join(base_dir, 'results', 'expert_ablation', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(seeds[0])
    graph_dataset = GraphDataset(args.dataset, args.data_path, svd_dim=args.svd_dim)
    task_loader = TaskLoader(
        batch_size=config.get('batch_size', 256),
        graph_dataset=graph_dataset,
        class_splits=exp['class_splits'],
        split_S=exp.get('split_S', 5),
        split_t=exp.get('split_t', 3),
        split_v=exp.get('split_v', 1),
    )

    all_data = {}
    for n_exp in expert_counts:
        print(f"\n{'#'*70}")
        print(f"  Running with max_experts = {n_exp}")
        print(f"{'#'*70}")

        result = run_with_experts(n_exp, task_loader, config, device, ntrials, seeds)
        all_data[n_exp] = result

        print(f"\n  [max_experts={n_exp}] "
              f"Final Joint Acc (micro): {result['joint_acc'][-1]:.4f}, "
              f"(macro): {result['joint_macro_acc'][-1]:.4f}")

    # Save raw results
    serializable = {}
    for n, d in all_data.items():
        serializable[str(n)] = {
            'joint_acc': [float(v) for v in d['joint_acc']],
            'joint_macro_acc': [float(v) for v in d['joint_macro_acc']],
            'acc_matrix': [[float(v) for v in row] for row in d['acc_matrix']],
        }
    with open(os.path.join(out_dir, 'expert_ablation.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved raw results: {os.path.join(out_dir, 'expert_ablation.json')}")

    # Generate plots
    num_sessions = len(exp['class_splits'])
    plot_lines(all_data, expert_counts, num_sessions, args.dataset, out_dir)
    plot_heatmaps(all_data, expert_counts, num_sessions, args.dataset, out_dir)
    plot_ablation_legend(expert_counts, out_dir)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"EXPERT ABLATION RESULTS ({args.dataset}, {ntrials} trial(s))")
    print(f"{'='*60}")
    print(f"{'Experts':<10} {'Final Micro':>12} {'Final Macro':>12}")
    print(f"{'-'*34}")
    for n_exp in expert_counts:
        micro = all_data[n_exp]['joint_acc'][-1]
        macro = all_data[n_exp]['joint_macro_acc'][-1]
        print(f"{n_exp:<10} {micro:>12.4f} {macro:>12.4f}")


if __name__ == '__main__':
    main()
