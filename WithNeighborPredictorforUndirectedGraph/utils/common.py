"""
Common utilities for the continual learning framework.
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
    """Load model checkpoint."""
    filepath = os.path.join(path, f"{dataset}_{model_name}_seed{seed}.pt")
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return True
    return False


class CLMetric:
    """
    Metric tracker for continual learning experiments.
    Tracks accuracy and forgetting across sessions.
    
    输出格式:
    1. Accuracy Matrix - 每个session测试所有任务的准确率
    2. Joint Accuracy Summary - 每个session测试所有已训练类别的准确率
    """
    
    def __init__(self):
        self.acc_matrix = []  # acc_matrix[session][task] = accuracy
        self.joint_acc = []   # Joint accuracy per session
    
    def add_results(self, acc_list, joint_acc):
        """
        Add results for a session.
        
        Args:
            acc_list: List of accuracies for each task (including current)
            joint_acc: Joint accuracy on all seen tasks
        """
        self.acc_matrix.append(acc_list)
        self.joint_acc.append(joint_acc)
    
    def get_results(self):
        """
        Compute final metrics.
        
        Returns:
            avg_acc: Average accuracy across all tasks
            avg_fgt: Average forgetting
            avg_joint_acc: Average joint accuracy
            last_joint_acc: Final joint accuracy
        """
        if len(self.acc_matrix) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        num_sessions = len(self.acc_matrix)
        
        # Average accuracy (mean of final session's per-task accuracies)
        final_acc_list = self.acc_matrix[-1]
        avg_acc = np.mean(final_acc_list)
        
        # Average forgetting
        forgetting_list = []
        for task_id in range(num_sessions - 1):
            # Best accuracy on this task (when it was trained)
            best_acc = self.acc_matrix[task_id][task_id]
            # Final accuracy on this task
            final_acc = self.acc_matrix[-1][task_id]
            forgetting = best_acc - final_acc
            forgetting_list.append(forgetting)
        
        avg_fgt = np.mean(forgetting_list) if forgetting_list else 0.0
        
        # Joint accuracy
        avg_joint_acc = np.mean(self.joint_acc)
        last_joint_acc = self.joint_acc[-1] if self.joint_acc else 0.0
        
        return avg_acc, avg_fgt, avg_joint_acc, last_joint_acc
    
    def print_matrix(self):
        """
        Print the accuracy matrix.
        
        Format:
        Accuracy Matrix:
        Session | Task 0 | Task 1 | Task 2 | Task 3 | Task 4
        --------------------------------------------------
           0    | 0.9300
           1    | 0.4800 | 0.6900
           2    | 0.2700 | 0.6200 | 0.6800
           3    | 0.2100 | 0.5900 | 0.5700 | 0.4800
           4    | 0.1400 | 0.5300 | 0.5200 | 0.5000 | 0.6369
        """
        if len(self.acc_matrix) == 0:
            print("No results to display.")
            return
        
        num_sessions = len(self.acc_matrix)
        
        # Header
        header = "Session | " + " | ".join([f"Task {i}" for i in range(num_sessions)])
        print("\nAccuracy Matrix:")
        print(header)
        print("-" * len(header))
        
        # Rows
        for session_id, acc_list in enumerate(self.acc_matrix):
            row_parts = []
            for task_id in range(num_sessions):
                if task_id < len(acc_list):
                    row_parts.append(f"{acc_list[task_id]:.4f}")
                else:
                    row_parts.append("      ")
            
            row = f"   {session_id}    | " + " | ".join(row_parts)
            print(row)
