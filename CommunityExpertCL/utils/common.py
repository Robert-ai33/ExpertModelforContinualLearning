"""
Common utilities for CommunityExpertCL.
"""

import os
import random
import torch
import numpy as np


def seed_everything(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, path, dataset, model_name, seed):
    """Save model checkpoint."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{dataset}_{model_name}_seed{seed}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(model, path, dataset, model_name, seed):
    """Load model checkpoint. Returns True if loaded."""
    filepath = os.path.join(path, f"{dataset}_{model_name}_seed{seed}.pt")
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return True
    return False


class CLMetric:
    """Accuracy and forgetting tracker for continual learning."""

    def __init__(self):
        self.acc_matrix = []
        self.joint_acc = []

    def add_results(self, acc_list, joint_acc):
        self.acc_matrix.append(acc_list)
        self.joint_acc.append(joint_acc)

    def get_results(self):
        if not self.acc_matrix:
            return 0.0, 0.0, 0.0, 0.0

        num_sessions = len(self.acc_matrix)
        final_acc_list = self.acc_matrix[-1]
        avg_acc = np.mean(final_acc_list)

        forgetting_list = []
        for task_id in range(num_sessions - 1):
            best_acc = self.acc_matrix[task_id][task_id]
            final_acc = self.acc_matrix[-1][task_id]
            forgetting_list.append(best_acc - final_acc)

        avg_fgt = np.mean(forgetting_list) if forgetting_list else 0.0
        avg_joint_acc = np.mean(self.joint_acc)
        last_joint_acc = self.joint_acc[-1] if self.joint_acc else 0.0

        return avg_acc, avg_fgt, avg_joint_acc, last_joint_acc

    def print_matrix(self):
        if not self.acc_matrix:
            print("No results to display.")
            return
        num_sessions = len(self.acc_matrix)
        header = "Session | " + " | ".join([f"Task {i}" for i in range(num_sessions)])
        print("\nAccuracy Matrix:")
        print(header)
        print("-" * len(header))
        for sid, acc_list in enumerate(self.acc_matrix):
            parts = []
            for tid in range(num_sessions):
                parts.append(f"{acc_list[tid]:.4f}" if tid < len(acc_list) else "      ")
            print(f"   {sid}    | " + " | ".join(parts))


class PurityMetric:
    """Cluster purity tracker."""

    def __init__(self):
        self.session_results = []

    def add_session_result(self, session_id, clusters_info, overall_purity):
        self.session_results.append({
            'session_id': session_id,
            'clusters_info': clusters_info,
            'overall_purity': overall_purity,
        })

    def print_results(self):
        for res in self.session_results:
            sid = res['session_id']
            info = res['clusters_info']
            purity = res['overall_purity']
            print(f"\n--- Session {sid} Clustering Results ---")
            print(f"Overall Purity: {purity:.4f}")

            large_clusters = [(i, c) for i, c in enumerate(info) if c['size'] > 5]
            small_clusters = [(i, c) for i, c in enumerate(info) if c['size'] <= 5]

            for i, c in large_clusters:
                print(f"  Cluster {i}: size={c['size']}, "
                      f"dominant_class={c['dominant_class']} "
                      f"(purity={c['purity']:.4f})")

            if small_clusters:
                size_counts = {}
                for _, c in small_clusters:
                    s = c['size']
                    size_counts[s] = size_counts.get(s, 0) + 1
                parts = [f"{cnt} clusters of size {s}" for s, cnt in sorted(size_counts.items())]
                print(f"  Small clusters (size<=5): {', '.join(parts)}")

    def get_summary(self):
        if not self.session_results:
            return 0.0, 0.0
        purities = [r['overall_purity'] for r in self.session_results]
        return np.mean(purities), purities[-1]
