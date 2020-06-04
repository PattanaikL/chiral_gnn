import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool


class Net(torch.nn.Module):
    def __init__(self, dim, num_node_features, num_edge_features):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(num_node_features, dim)

        mlp = nn.Sequential(nn.Linear(num_edge_features, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, mlp, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = global_mean_pool(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)