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
from models import (
    LiteExpertCL, BaselineCL, SEEDCL, MAERoutingOnlyCL, ACILCL, TEMCL,
    DINGLECL,
)
from utils import seed_everything, compute_ap_af

sys.path.insert(0, os.path.dirname(__file__))

# Import EXP_SETTINGS from main
from main import EXP_SETTINGS

ALL_METHODS = ['bare', 'ewc', 'mas', 'twp', 'lwf', 'gem', 'ergnn', 'cat',
               'cosine', 'teen', 'delome', 'seed', 'acil', 'tem', 'dingle',
               'joint', 'mae_routing', 'lite']
STANDALONE_METHODS = {'lite', 'seed', 'mae_routing', 'acil', 'tem', 'dingle'}
SUPPORTED_RUN_METHODS = [m for m in ALL_METHODS
                        if m in STANDALONE_METHODS or m in BaselineCL.METHODS]
METHOD_LABELS = {
    'bare': 'BARE', 'ewc': 'EWC', 'mas': 'MAS',
    'twp': 'TWP', 'lwf': 'LwF', 'gem': 'GEM',
    'ergnn': 'ER-GNN', 'cat': 'CaT',
    'cosine': 'COSINE', 'teen': 'TEEN',
    'delome': 'DeLoMe', 'seed': 'SEED',
    'acil': 'ACIL', 'tem': 'TEM',
    'dingle': 'DINGLE',
    'joint': 'JOINT', 'lite': 'Ours',
    'mae_routing': 'Manifold_MAE',
}
METHOD_MARKERS = {
    'bare': 'o', 'ewc': 's', 'mas': '^',
    'twp': 'v', 'lwf': 'D', 'gem': 'P',
    'ergnn': '*', 'cat': 'X',
    'cosine': 'p', 'teen': 'h',
    'delome': '>', 'seed': '8',
    'acil': '+', 'tem': '2',
    'dingle': '3',
    'joint': '<', 'lite': 'd',
    'mae_routing': '1',
}


def run_single_method(method, task_loader, configs, device, ntrials, seeds):
    """Run one method for multiple trials, return averaged results.

    ``configs`` is a dict keyed by standalone-method name (plus 'baseline' for
    the shared BaselineCL/SEED config), e.g.
        {'lite': ..., 'baseline': ..., 'mae_routing': ..., 'acil': ...}
    """
    all_results = []
    for trial in range(ntrials):
        seed = seeds[trial]
        seed_everything(seed)

        if method == 'lite':
            model = LiteExpertCL(
                task_loader=task_loader, config=configs['lite'], device=device)
        elif method == 'mae_routing':
            model = MAERoutingOnlyCL(
                task_loader=task_loader, config=configs['mae_routing'],
                device=device)
        elif method == 'acil':
            model = ACILCL(
                task_loader=task_loader, config=configs['acil'], device=device)
        elif method == 'tem':
            model = TEMCL(
                task_loader=task_loader, config=configs['tem'], device=device)
        elif method == 'dingle':
            model = DINGLECL(
                task_loader=task_loader, config=configs['dingle'], device=device)
        elif method == 'seed':
            model = SEEDCL(
                task_loader=task_loader, config=configs['baseline'],
                device=device)
        else:
            model = BaselineCL(
                task_loader=task_loader, config=configs['baseline'],
                device=device, method=method)

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

    # AP / AF (CGLB convention): compute on the trial-averaged acc matrix so
    # the reported numbers match the heatmap exactly. Per-trial std is also
    # reported below for transparency.
    ap_history, af, final_ap = compute_ap_af(avg_matrix)
    per_trial_final_ap = [compute_ap_af(r['acc_matrix'])[2] for r in all_results]
    per_trial_af = [compute_ap_af(r['acc_matrix'])[1] for r in all_results]
    final_ap_std = float(np.std(per_trial_final_ap)) if len(per_trial_final_ap) > 1 else 0.0
    af_std = float(np.std(per_trial_af)) if len(per_trial_af) > 1 else 0.0

    return {
        'joint_acc': avg_joint_acc,
        'joint_macro_acc': avg_joint_macro,
        'acc_matrix': avg_matrix,
        'ap_history': ap_history,
        'af': af,
        'final_ap': final_ap,
        'final_ap_std': final_ap_std,
        'af_std': af_std,
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
    args = parser.parse_args()

    if args.methods == 'all':
        methods = SUPPORTED_RUN_METHODS
    else:
        methods = [m.strip() for m in args.methods.split(',')]
        unsupported = [m for m in methods
                       if m not in STANDALONE_METHODS and m not in BaselineCL.METHODS]
        if unsupported:
            raise ValueError(
                f"Methods not yet runnable in this script: {unsupported}. "
                f"Supported methods: {SUPPORTED_RUN_METHODS}"
            )

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    exp = EXP_SETTINGS[args.dataset]

    # Load all standalone-method configs + the shared baseline config.
    base_dir = os.path.dirname(__file__)
    cfg_files = {
        'lite': 'config_lite.yaml',
        'baseline': 'config_baseline.yaml',
        'mae_routing': 'config_mae_routing.yaml',
        'acil': 'config_acil.yaml',
        'tem': 'config_tem.yaml',
        'dingle': 'config_dingle.yaml',
    }
    configs = {}
    for key, fname in cfg_files.items():
        with open(os.path.join(base_dir, 'configs', fname),
                  'r', encoding='utf-8') as f:
            configs[key] = yaml.safe_load(f)['default']

    for cfg in configs.values():
        cfg['class_splits'] = exp['class_splits']
        cfg['split_S'] = exp.get('split_S', cfg.get('split_S', 5))
        cfg['split_t'] = exp.get('split_t', cfg.get('split_t', 3))
        cfg['split_v'] = exp.get('split_v', cfg.get('split_v', 1))

    seeds = configs['lite'].get('seed', [0, 1, 2, 3, 4])
    ntrials = min(args.ntrials, len(seeds))

    out_dir = os.path.join(base_dir, 'results', 'comparison', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(seeds[0])
    graph_dataset = GraphDataset(args.dataset, args.data_path, svd_dim=args.svd_dim)
    task_loader = TaskLoader(
        batch_size=configs['lite'].get('batch_size', 256),
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

        result = run_single_method(method, task_loader, configs,
                                   device, ntrials, seeds)
        all_data[method] = result

        print(f"\n  [{METHOD_LABELS.get(method, method)}] "
              f"Final Joint Acc (micro): {result['joint_acc'][-1]:.4f}, "
              f"(macro): {result['joint_macro_acc'][-1]:.4f}, "
              f"AP: {result['final_ap']:.4f}, "
              f"AF: {result['af']:+.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    # Save raw results
    serializable = {}
    for m, d in all_data.items():
        serializable[m] = {
            'joint_acc': [float(v) for v in d['joint_acc']],
            'joint_macro_acc': [float(v) for v in d['joint_macro_acc']],
            'acc_matrix': [[float(v) for v in row] for row in d['acc_matrix']],
            'ap_history': [float(v) for v in d.get('ap_history', [])],
            'af': float(d.get('af', 0.0)),
            'final_ap': float(d.get('final_ap', 0.0)),
            'final_ap_std': float(d.get('final_ap_std', 0.0)),
            'af_std': float(d.get('af_std', 0.0)),
        }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved raw results: {os.path.join(out_dir, 'results.json')}")

    # Generate plots
    num_sessions = len(exp['class_splits'])
    plot_line_charts(all_data, num_sessions, args.dataset, out_dir)
    plot_heatmaps(all_data, num_sessions, args.dataset, out_dir)
    plot_shared_legend(out_dir)

    # Print final comparison table (including CGLB AP / AF)
    show_std = ntrials > 1
    if show_std:
        col_widths = (20, 14, 14, 18, 18)
        header = (f"{'Method':<{col_widths[0]}} "
                  f"{'Joint Micro':>{col_widths[1]}} "
                  f"{'Joint Macro':>{col_widths[2]}} "
                  f"{'AP (mean +/- std)':>{col_widths[3]}} "
                  f"{'AF (mean +/- std)':>{col_widths[4]}}")
    else:
        col_widths = (20, 14, 14, 12, 12)
        header = (f"{'Method':<{col_widths[0]}} "
                  f"{'Joint Micro':>{col_widths[1]}} "
                  f"{'Joint Macro':>{col_widths[2]}} "
                  f"{'AP':>{col_widths[3]}} "
                  f"{'AF':>{col_widths[4]}}")
    width = sum(col_widths) + len(col_widths)

    print(f"\n{'=' * max(70, width)}")
    print(f"COMPARISON TABLE ({args.dataset}, {ntrials} trial(s))")
    print("CL Matrix: CGLB protocol (cell (k,t) = subgraph_per_task[k] eval on test_idx[t])")
    print(f"{'=' * max(70, width)}")
    print(header)
    print('-' * len(header))
    for method in methods:
        if method not in all_data:
            continue
        label = METHOD_LABELS.get(method, method)
        d = all_data[method]
        micro = d['joint_acc'][-1]
        macro = d['joint_macro_acc'][-1]
        ap = d.get('final_ap', 0.0)
        af = d.get('af', 0.0)
        if show_std:
            ap_str = f"{ap:.4f} +/- {d.get('final_ap_std', 0.0):.4f}"
            af_str = f"{af:+.4f} +/- {d.get('af_std', 0.0):.4f}"
            print(f"{label:<{col_widths[0]}} "
                  f"{micro:>{col_widths[1]}.4f} "
                  f"{macro:>{col_widths[2]}.4f} "
                  f"{ap_str:>{col_widths[3]}} "
                  f"{af_str:>{col_widths[4]}}")
        else:
            print(f"{label:<{col_widths[0]}} "
                  f"{micro:>{col_widths[1]}.4f} "
                  f"{macro:>{col_widths[2]}.4f} "
                  f"{ap:>{col_widths[3]}.4f} "
                  f"{af:>+{col_widths[4]}.4f}")


if __name__ == '__main__':
    main()
