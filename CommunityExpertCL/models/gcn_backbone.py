"""
GCN backbone for baseline continual learning methods.
Uses PyG's GCNConv with configurable layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNBackbone(nn.Module):
    """Multi-layer GCN with configurable depth."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, output_dim))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, output_dim))

        self.second_last_h = None

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        self.second_last_h = h
        logits = self.convs[-1](h, edge_index)
        return logits

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
