import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class Net(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Net, self).__init__()
        self.feature_pre = nn.Linear(num_node_features, 300)
        self.conv = GCNConv(300, 300)
        self.readout = nn.Linear(300, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.feature_pre(x))
        for _ in range(3):
            x = self.conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)
        return self.readout(x).squeeze(-1)
