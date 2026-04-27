"""
GCN Baseline with Knowledge Distillation for Class-Incremental Continual Learning

Model structure:
- Single GCN layer + BatchNorm + ReLU + Dropout + Linear classifier

Training strategy:
- Train on undirected graph (derived from directed graph)
- Each session only trains on current session's class nodes
- Knowledge distillation (LwF-style): at session > 0, a frozen copy of the
  old model is kept. KL divergence on previous classes' logits is added to
  the CE loss to preserve old knowledge.

Loss = CE(current classes) + λ * KL(previous classes distribution)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from torch_geometric.nn import GCNConv
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from utils import save_checkpoint, load_checkpoint


class NaiveGCNModel(nn.Module):
    """
    GCN model: GCN + BatchNorm + ReLU + Dropout + Linear classifier.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv = GCNConv(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def get_embeddings(self, x, edge_index):
        """Get GCN-encoded embeddings."""
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        """Forward pass, returns logits and embeddings."""
        embeddings = self.get_embeddings(x, edge_index)
        logits = self.classifier(embeddings)
        return logits, embeddings


class NaiveGCNCL:
    """
    GCN + KD for Class-Incremental Continual Learning.

    Key characteristics:
    - Single GCN + BN + Linear model
    - Each session: CE on current classes + KL distillation on previous classes
    """

    def __init__(self, task_loader, result_logger, config, checkpoint_path,
                 seed, device):
        self.task_loader = task_loader
        self.result_logger = result_logger
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.device = device

        # Model parameters
        self.input_dim = task_loader.data.x.shape[1]
        self.num_classes = task_loader.data.y.max().item() + 1
        self.hidden_dim = config['hidden_dim']
        self.dropout = config['dropout']

        # Training parameters
        self.epochs = config['epochs']
        self.debug = config.get('debug', False)

        # Knowledge distillation parameters
        self.kd_lambda = config.get('kd_lambda', 1.0)
        self.kd_temperature = config.get('kd_temperature', 2.0)

        # Create model
        self.model = NaiveGCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=self.dropout,
        ).to(device)

        self.old_model = None
        self.current_session = 0

    # ======================== Training Functions ========================

    def train_epoch(self, subgraph, train_loader, optimizer,
                    curr_classes, all_classes, prev_classes):
        """
        Train for one epoch on undirected graph.

        Loss = CE(current classes) + λ * KL(previous classes distribution)
        """
        self.model.train()

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)

        total_loss = 0.0
        num_samples = 0

        # Precompute old model logits if distillation is active
        old_logits_all = None
        if self.old_model is not None and len(prev_classes) > 0:
            self.old_model.eval()
            with torch.no_grad():
                old_logits_all, _ = self.old_model(x, edge_index)

        for batch in train_loader:
            optimizer.zero_grad()

            logits, _ = self.model(x, edge_index)

            node_ids = batch['node_id']
            labels = batch['labels'].to(self.device)
            batch_logits = logits[node_ids]

            max_class = max(all_classes) + 1
            batch_logits = batch_logits[:, :max_class]

            # CE loss: isolated softmax on current session's classes only
            valid_mask = torch.zeros(labels.size(0), dtype=torch.bool,
                                     device=self.device)
            for c in curr_classes:
                valid_mask |= (labels == c)

            if valid_mask.sum() == 0:
                continue

            curr_idx = torch.tensor(curr_classes, dtype=torch.long,
                                    device=self.device)
            curr_logits = batch_logits[valid_mask][:, curr_idx]
            local_labels = torch.zeros_like(labels[valid_mask])
            for i, c in enumerate(curr_classes):
                local_labels[labels[valid_mask] == c] = i
            ce_loss = F.cross_entropy(curr_logits, local_labels)

            # KL distillation on previous classes only
            loss = ce_loss
            if old_logits_all is not None:
                prev_idx = torch.tensor(prev_classes, dtype=torch.long,
                                        device=self.device)
                old_batch_logits = old_logits_all[node_ids]

                new_prev = batch_logits[:, prev_idx] / self.kd_temperature
                old_prev = old_batch_logits[:, prev_idx] / self.kd_temperature

                p_new = F.log_softmax(new_prev, dim=-1)
                p_old = F.softmax(old_prev, dim=-1)

                kd_loss = F.kl_div(p_new, p_old, reduction='batchmean')
                loss = ce_loss + self.kd_lambda * kd_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * valid_mask.sum().item()
            num_samples += valid_mask.sum().item()

        return total_loss / num_samples if num_samples > 0 else 0.0

    # ======================== Validation Functions ========================

    @torch.no_grad()
    def evaluate_classifier(self, subgraph, data_loader,
                            eval_classes, all_classes):
        """Validate classifier on given evaluation classes."""
        self.model.eval()

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)

        all_preds = []
        all_labels = []

        for batch in data_loader:
            node_ids = batch['node_id']
            labels = batch['labels'].to(self.device)

            logits, _ = self.model(x, edge_index)
            batch_logits = logits[node_ids]

            max_class = max(all_classes) + 1
            batch_logits = batch_logits[:, :max_class]

            valid_mask = torch.zeros(labels.size(0), dtype=torch.bool,
                                     device=self.device)
            for c in eval_classes:
                valid_mask |= (labels == c)

            if valid_mask.sum() == 0:
                continue

            preds = torch.argmax(batch_logits[valid_mask], dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels[valid_mask].cpu())

        if len(all_preds) == 0:
            return 0.0

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        return accuracy_score(labels, preds)

    # ======================== Evaluation ========================

    @torch.no_grad()
    def evaluate(self, subgraph, test_loader, trained_classes):
        """
        Evaluate on test set using the single model directly.

        No expert selection needed - just forward pass and argmax.
        """
        self.model.eval()

        x = subgraph['x'].to(self.device)
        edge_index = subgraph['edge_index'].to(self.device)

        all_preds = []
        all_labels = []

        for batch in test_loader:
            node_ids = batch['node_id']
            labels = batch['labels']

            logits, _ = self.model(x, edge_index)
            batch_logits = logits[node_ids]

            max_class = max(trained_classes) + 1
            batch_logits = batch_logits[:, :max_class]

            preds = torch.argmax(batch_logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

        if len(all_preds) == 0:
            return 0.0, 0.0

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return acc, f1

    # ======================== Main Training Loop ========================

    def fit(self, trial):
        """
        Main training loop for GCN + KD.

        For each session:
        1. Freeze old model (session > 0) for distillation
        2. Train: CE on current classes + KL on previous classes
        3. Evaluate on isolated / joint / previous test sets
        """
        joint_acc_history = []

        for session_id in range(self.task_loader.sessions):
            self.current_session = session_id

            # Load checkpoint from previous session
            if session_id > 0:
                load_checkpoint(self.model, self.checkpoint_path,
                                'NaiveGCN', self.seed)

            # Get task data
            (curr_classes, all_classes,
             subgraph, joint_subgraph,
             train_loader, valid_loader,
             test_loader_iso, test_loader_joint) = self.task_loader.get_task(session_id)

            prev_classes = sorted(set(all_classes) - set(curr_classes))

            # Freeze old model for distillation before training new session
            if session_id > 0:
                self.old_model = copy.deepcopy(self.model)
                self.old_model.to(self.device)
                for p in self.old_model.parameters():
                    p.requires_grad = False
                self.old_model.eval()

            print(f"\n{'='*60}")
            print(f"Session {session_id}: Classes {curr_classes}")
            print(f"All classes so far: {all_classes}")
            print(f"KD prev classes: {prev_classes if prev_classes else 'N/A (session 0)'}")
            print(f"Model: NaiveGCN + KD (lambda={self.kd_lambda}, "
                  f"T={self.kd_temperature})")
            print(f"{'='*60}")

            # ========== Training ==========
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(self.config['lr']),
                weight_decay=float(self.config['weight_decay'])
            )
            best_acc = 0.0
            patience_counter = 0

            pbar = tqdm(range(self.epochs),
                        desc=f"Session {session_id} - NaiveGCN+KD")
            for epoch in pbar:
                loss = self.train_epoch(
                    subgraph, train_loader, optimizer,
                    curr_classes, all_classes, prev_classes
                )

                if epoch > 0 and epoch % self.config['valid_epoch'] == 0:
                    valid_acc = self.evaluate_classifier(
                        subgraph, valid_loader, curr_classes, all_classes
                    )
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        patience_counter = 0
                        save_checkpoint(self.model, optimizer, epoch,
                                        self.checkpoint_path,
                                        'NaiveGCN', self.seed)
                    else:
                        patience_counter += 1
                        if patience_counter > self.config['patience']:
                            break

                    pbar.set_postfix({'loss': f'{loss:.4f}',
                                      'val_acc': f'{valid_acc:.4f}'})
                else:
                    pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Load best model
            load_checkpoint(self.model, self.checkpoint_path,
                            'NaiveGCN', self.seed)

            print(f"\n  Best validation accuracy: {best_acc:.4f}")

            # ========== Testing ==========
            test_acc_iso, _ = self.evaluate(
                subgraph, test_loader_iso, all_classes
            )
            test_acc_joint, _ = self.evaluate(
                joint_subgraph, test_loader_joint, all_classes
            )

            # Evaluate previous tasks
            acc_list = []
            for prev_session in range(session_id):
                prev_subgraph = self.task_loader.subgraph_per_task[prev_session]
                prev_test_idx = self.task_loader.test_idx_per_task[prev_session]
                prev_test_dataset = torch.utils.data.Subset(
                    self.task_loader.dataset, prev_test_idx
                )
                prev_test_loader = torch.utils.data.DataLoader(
                    prev_test_dataset,
                    batch_size=self.config['batch_size'], shuffle=False
                )
                prev_acc, _ = self.evaluate(
                    prev_subgraph, prev_test_loader, all_classes
                )
                acc_list.append(prev_acc)
            acc_list.append(test_acc_iso)

            joint_acc_history.append({
                'session': session_id,
                'classes': all_classes.copy(),
                'joint_acc': test_acc_joint
            })

            print(f"\nSession {session_id} Results:")
            print(f"  Isolate Acc: {test_acc_iso:.4f}")
            print(f"  Joint Acc (all {len(all_classes)} classes): "
                  f"{test_acc_joint:.4f}")

            self.result_logger.add_results(acc_list, test_acc_joint)

        # Print final results
        self.result_logger.print_matrix()

        print(f"\n{'='*60}")
        print("Joint Accuracy Summary")
        print(f"{'='*60}")
        for record in joint_acc_history:
            print(f"  Session {record['session']}: "
                  f"Classes {record['classes']} -> "
                  f"Acc: {record['joint_acc']:.4f}")
        print(f"{'='*60}\n")

        return self.result_logger
