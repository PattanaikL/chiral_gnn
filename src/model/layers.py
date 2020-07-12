from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(MessagePassing):
    def __init__(self, args):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.batch_norm = nn.BatchNorm1d(args.hidden_size)

    def forward(self, x, edge_index, edge_attr):
        # no edge updates
        x = self.linear(x)

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)
        return self.batch_norm(x), edge_attr

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class GINEConv(MessagePassing):
    def __init__(self, args):
        super(GINEConv, self).__init__(aggr="add")
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.mlp = nn.Sequential(nn.Linear(args.hidden_size, 2 * args.hidden_size),
                                 nn.BatchNorm1d(2 * args.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(2 * args.hidden_size, args.hidden_size))
        self.batch_norm = nn.BatchNorm1d(args.hidden_size)

    def forward(self, x, edge_index, edge_attr):
        # no edge updates
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.mlp((1 + self.eps) * x + x_new)
        return self.batch_norm(x), edge_attr

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


class DMPNNConv(MessagePassing):
    def __init__(self, args):
        super(DMPNNConv, self).__init__(aggr='add')
        self.mlp = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                 nn.BatchNorm1d(args.hidden_size),
                                 nn.ReLU())

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)
        rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)
        return x, self.mlp(a_message[row] - rev_message)

    def message(self, x_j, edge_attr):
        return edge_attr
