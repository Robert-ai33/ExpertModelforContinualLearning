"""
Run all CL methods (baselines + LiteExpertCL) and generate comparison plots.

Usage:
  python run_all.py --dataset cora --gpu 0
  python run_all.py --dataset coauthor-cs --gpu 0 --ntrials 3

Outputs:
  results/comparison/<dataset>/joint_micro.png   - Joint micro accuracy line plot
  results/comparison/<dataset>/joint_macro.png   - Joint macro accuracy line plot
  results/comparison/<dataset>/heatmaps.png      - Accuracy matrix heatmaps
  results/comparison/legend.png                  - Shared legend for all datasets
"""

import os
import sys
import gc
import argparse
import yaml
import json
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import GraphDataset, TaskLoader
from models import LiteExpertCL, BaselineCL
from utils import seed_everything

sys.path.insert(0, os.path.dirname(__file__))

# Import EXP_SETTINGS from main
from main import EXP_SETTINGS

ALL_METHODS = ['bare', 'ewc', 'mas', 'twp', 'lwf', 'gem', 'ergnn', 'cat',
               'cosine', 'teen', 'delome', 'joint', 'lite']
SUPPORTED_RUN_METHODS = [m for m in ALL_METHODS if m == 'lite' or m in BaselineCL.METHODS]
METHOD_LABELS = {
    'bare': 'BARE', 'ewc': 'EWC', 'mas': 'MAS',
    'twp': 'TWP', 'lwf': 'LwF', 'gem': 'GEM',
    'ergnn': 'ER-GNN', 'cat': 'CaT',
    'cosine': 'COSINE', 'teen': 'TEEN',
    'delome': 'DeLoMe',
    'joint': 'JOINT', 'lite': 'Ours',
}
METHOD_MARKERS = {
    'bare': 'o', 'ewc': 's', 'mas': '^',
    'twp': 'v', 'lwf': 'D', 'gem': 'P',
    'ergnn': '*', 'cat': 'X',
    'cosine': 'p', 'teen': 'h',
    'delome': '>', 'joint': '<', 'lite': 'd',
}


def run_single_method(method, task_loader, config_lite, config_baseline, device, ntrials, seeds):
    """Run one method for multiple trials, return averaged results."""
    all_results = []
    for trial in range(ntrials):
        seed = seeds[trial]
        seed_everything(seed)

        if method == 'lite':
            model = LiteExpertCL(task_loader=task_loader, config=config_lite, device=device)
        else:
            model = BaselineCL(task_loader=task_loader, config=config_baseline, device=device, method=method)

        results = model.fit(trial)
        all_results.append(results)

    num_sessions = len(all_results[0]['joint_acc'])
    avg_joint_acc = [np.mean([r['joint_acc'][s] for r in all_results]) for s in range(num_sessions)]
    avg_joint_macro = [np.mean([r['joint_macro_acc'][s] for r in all_results]) for s in range(num_sessions)]

    avg_matrix = []
    for s in range(num_sessions):
        row_len = s + 1
        avg_row = []
        for t in range(row_len):
            vals = [r['acc_matrix'][s][t] for r in all_results]
            avg_row.append(np.mean(vals))
        avg_matrix.append(avg_row)

    return {
        'joint_acc': avg_joint_acc,
        'joint_macro_acc': avg_joint_macro,
        'acc_matrix': avg_matrix,
    }


def plot_line_charts(all_data, num_sessions, dataset, out_dir):
    """Generate joint accuracy line plots (no title, no ylabel, no legend)."""
    sessions = list(range(num_sessions))

    for metric, fname in [
        ('joint_acc', 'joint_micro.png'),
        ('joint_macro_acc', 'joint_macro.png'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in ALL_METHODS:
            if method not in all_data:
                continue
            label = METHOD_LABELS.get(method, method)
            mkr = METHOD_MARKERS.get(method, 'o')
            vals = all_data[method][metric]
            ax.plot(sessions, vals, marker=mkr, markersize=8, label=label, linewidth=1.8)

        ax.set_xlabel('Session', fontsize=13)
        ax.set_xticks(sessions)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
        print(f"Saved: {os.path.join(out_dir, fname)}")


def plot_shared_legend(out_dir):
    """Generate a shared legend image for all comparison datasets."""
    legend_dir = os.path.dirname(out_dir)
    fig, ax = plt.subplots(figsize=(12, 1))
    for method in ALL_METHODS:
        label = METHOD_LABELS.get(method, method)
        mkr = METHOD_MARKERS.get(method, 'o')
        ax.plot([], [], marker=mkr, markersize=8, label=label, linewidth=1.8)
    ax.legend(fontsize=10, loc='center', ncol=len(ALL_METHODS), frameon=False)
    ax.axis('off')
    fig.tight_layout()
    legend_path = os.path.join(legend_dir, 'legend.png')
    fig.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved shared legend: {legend_path}")


def plot_heatmaps(all_data, num_sessions, dataset, out_dir):
    """Generate accuracy matrix heatmaps in a grid layout."""
    methods_to_plot = [m for m in ALL_METHODS if m in all_data]
    n = len(methods_to_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, method in enumerate(methods_to_plot):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        mat = all_data[method]['acc_matrix']

        full_mat = np.full((num_sessions, num_sessions), np.nan)
        for s, row in enumerate(mat):
            for t, val in enumerate(row):
                full_mat[s][t] = val * 100

        masked = np.ma.array(full_mat, mask=np.isnan(full_mat))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')

        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect='equal', origin='upper')
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=13, fontweight='bold')
        ax.set_xlabel('Tasks', fontsize=10)
        ax.set_ylabel('Tasks', fontsize=10)
        ax.set_xticks(range(num_sessions))
        ax.set_yticks(range(num_sessions))

        if num_sessions <= 10:
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

    fig.suptitle(f'{dataset} - CL Accuracy Heatmaps', fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'heatmaps.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {os.path.join(out_dir, 'heatmaps.png')}")


def main():
    parser = argparse.ArgumentParser(description='Run all CL methods and compare')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=list(EXP_SETTINGS.keys()))
    parser.add_argument('--data_path', type=str, default='./data_files/')
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--methods', type=str, default='all',
                        help='Comma-separated methods or "all"')
    parser.add_argument('--svd_dim', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    if args.methods == 'all':
        methods = SUPPORTED_RUN_METHODS
    else:
        methods = [m.strip() for m in args.methods.split(',')]
        unsupported = [m for m in methods if m != 'lite' and m not in BaselineCL.METHODS]
        if unsupported:
            raise ValueError(
                f"Methods not yet runnable in this script: {unsupported}. "
                f"Supported methods: {SUPPORTED_RUN_METHODS}"
            )

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    exp = EXP_SETTINGS[args.dataset]

    # Load both configs
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, 'configs', 'config_lite.yaml'), 'r', encoding='utf-8') as f:
        config_lite = yaml.safe_load(f)['default']
    with open(os.path.join(base_dir, 'configs', 'config_baseline.yaml'), 'r', encoding='utf-8') as f:
        config_baseline = yaml.safe_load(f)['default']

    for cfg in [config_lite, config_baseline]:
        cfg['class_splits'] = exp['class_splits']
        cfg['split_S'] = exp.get('split_S', cfg.get('split_S', 5))
        cfg['split_t'] = exp.get('split_t', cfg.get('split_t', 3))
        cfg['split_v'] = exp.get('split_v', cfg.get('split_v', 1))
    config_lite['use_amp'] = args.amp

    seeds = config_lite.get('seed', [0, 1, 2, 3, 4])
    ntrials = min(args.ntrials, len(seeds))

    out_dir = os.path.join(base_dir, 'results', 'comparison', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(seeds[0])
    graph_dataset = GraphDataset(args.dataset, args.data_path, svd_dim=args.svd_dim)
    task_loader = TaskLoader(
        batch_size=config_lite.get('batch_size', 256),
        graph_dataset=graph_dataset,
        class_splits=exp['class_splits'],
        split_S=exp.get('split_S', 5),
        split_t=exp.get('split_t', 3),
        split_v=exp.get('split_v', 1),
    )

    all_data = {}
    for method in methods:
        print(f"\n{'#'*70}")
        print(f"  Running method: {METHOD_LABELS.get(method, method)}")
        print(f"{'#'*70}")

        result = run_single_method(method, task_loader, config_lite, config_baseline,
                                   device, ntrials, seeds)
        all_data[method] = result

        print(f"\n  [{METHOD_LABELS.get(method, method)}] "
              f"Final Joint Acc (micro): {result['joint_acc'][-1]:.4f}, "
              f"(macro): {result['joint_macro_acc'][-1]:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    # Save raw results
    serializable = {}
    for m, d in all_data.items():
        serializable[m] = {
            'joint_acc': [float(v) for v in d['joint_acc']],
            'joint_macro_acc': [float(v) for v in d['joint_macro_acc']],
            'acc_matrix': [[float(v) for v in row] for row in d['acc_matrix']],
        }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved raw results: {os.path.join(out_dir, 'results.json')}")

    # Generate plots
    num_sessions = len(exp['class_splits'])
    plot_line_charts(all_data, num_sessions, args.dataset, out_dir)
    plot_heatmaps(all_data, num_sessions, args.dataset, out_dir)
    plot_shared_legend(out_dir)

    # Print final comparison table
    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE ({args.dataset}, {ntrials} trial(s))")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Final Micro':>12} {'Final Macro':>12}")
    print(f"{'-'*44}")
    for method in methods:
        if method in all_data:
            label = METHOD_LABELS.get(method, method)
            micro = all_data[method]['joint_acc'][-1]
            macro = all_data[method]['joint_macro_acc'][-1]
            print(f"{label:<20} {micro:>12.4f} {macro:>12.4f}")


if __name__ == '__main__':
    main()
