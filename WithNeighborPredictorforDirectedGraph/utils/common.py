"""
Common utilities for the directed expert continual learning framework.
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


def save_checkpoint(model, optimizer, epoch, path, model_name, seed):
    """Save model checkpoint."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"wikics_{model_name}_seed{seed}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(model, path, model_name, seed):
    """Load model checkpoint."""
    filepath = os.path.join(path, f"wikics_{model_name}_seed{seed}.pt")
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return True
    return False


class CLMetric:
    """
    Metric tracker for continual learning experiments.
    Tracks accuracy and forgetting across sessions.
    """

    def __init__(self):
        self.acc_matrix = []
        self.joint_acc = []

    def add_results(self, acc_list, joint_acc):
        self.acc_matrix.append(acc_list)
        self.joint_acc.append(joint_acc)

    def get_results(self):
        if len(self.acc_matrix) == 0:
            return 0.0, 0.0, 0.0, 0.0

        num_sessions = len(self.acc_matrix)
        final_acc_list = self.acc_matrix[-1]
        avg_acc = np.mean(final_acc_list)

        forgetting_list = []
        for task_id in range(num_sessions - 1):
            best_acc = self.acc_matrix[task_id][task_id]
            final_acc = self.acc_matrix[-1][task_id]
            forgetting = best_acc - final_acc
            forgetting_list.append(forgetting)

        avg_fgt = np.mean(forgetting_list) if forgetting_list else 0.0
        avg_joint_acc = np.mean(self.joint_acc)
        last_joint_acc = self.joint_acc[-1] if self.joint_acc else 0.0

        return avg_acc, avg_fgt, avg_joint_acc, last_joint_acc

    def print_matrix(self):
        if len(self.acc_matrix) == 0:
            print("No results to display.")
            return

        num_sessions = len(self.acc_matrix)
        header = "Session | " + " | ".join([f"Task {i}" for i in range(num_sessions)])
        print("\nAccuracy Matrix:")
        print(header)
        print("-" * len(header))

        for session_id, acc_list in enumerate(self.acc_matrix):
            row_parts = []
            for task_id in range(num_sessions):
                if task_id < len(acc_list):
                    row_parts.append(f"{acc_list[task_id]:.4f}")
                else:
                    row_parts.append("      ")
            row = f"   {session_id}    | " + " | ".join(row_parts)
            print(row)
