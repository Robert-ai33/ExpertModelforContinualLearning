"""
Utility functions for SpectralClusteringCL.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, path, dataset, model_name, seed):
    """Save model checkpoint."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f'{dataset}_{model_name}_seed{seed}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(model, path, dataset, model_name, seed):
    """Load model checkpoint."""
    filepath = os.path.join(path, f'{dataset}_{model_name}_seed{seed}.pt')
    if not os.path.exists(filepath):
        return False
    checkpoint = torch.load(filepath, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return True


class PurityMetric:
    """
    Metric tracker for spectral clustering purity in continual learning.

    Records per-cluster purity at each session, and computes overall purity.
    """

    def __init__(self):
        self.session_results = []  # list of dicts per session

    def add_session_result(self, session_id, all_classes, cluster_purities, overall_purity):
        """
        Record results for one session.

        Args:
            session_id: session index
            all_classes: list of all classes seen so far
            cluster_purities: list of (cluster_id, purity, size, dominant_class) tuples
            overall_purity: weighted average purity
        """
        self.session_results.append({
            'session': session_id,
            'all_classes': all_classes,
            'cluster_purities': cluster_purities,
            'overall_purity': overall_purity,
        })

    def print_results(self):
        """Print all session results."""
        print(f"\n{'='*70}")
        print("Spectral Clustering Purity Results")
        print(f"{'='*70}")

        for result in self.session_results:
            session_id = result['session']
            all_classes = result['all_classes']
            overall_purity = result['overall_purity']

            print(f"\nSession {session_id} (Classes: {all_classes}):")
            print(f"  Overall Purity: {overall_purity:.4f}")
            cluster_purities = result['cluster_purities']
            print(f"  Per-cluster details ({len(cluster_purities)} communities):")

            small_clusters = {}
            for cluster_id, purity, size, dom_class in cluster_purities:
                if size > 5:
                    print(f"    Cluster {cluster_id}: purity={purity:.4f}, "
                          f"size={size}, dominant_class={dom_class}")
                elif size > 0:
                    small_clusters[size] = small_clusters.get(size, 0) + 1
            if small_clusters:
                parts = [f"{count}个{sz}节点社区"
                         for sz, count in sorted(small_clusters.items())]
                print(f"    小社区汇总: {', '.join(parts)}")

        print(f"\n{'='*70}")

        # Summary
        purities = [r['overall_purity'] for r in self.session_results]
        if purities:
            print(f"Purity across sessions: {[f'{p:.4f}' for p in purities]}")
            print(f"Average Purity: {np.mean(purities):.4f}")
            print(f"Last Session Purity: {purities[-1]:.4f}")
        print(f"{'='*70}\n")

    def get_summary(self):
        """Return summary statistics."""
        purities = [r['overall_purity'] for r in self.session_results]
        if not purities:
            return 0.0, 0.0
        return np.mean(purities), purities[-1]
