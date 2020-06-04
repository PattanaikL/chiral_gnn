from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class Net(torch.nn.Module):
    def __init__(self, num_node_features):
        super(Net, self).__init__()
        self.feature_pre = nn.Linear(num_node_features, 20)
        self.conv1 = GATConv(20, 20, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 20, 100, heads=1, dropout=0.6)

        self.readout = nn.Linear(100, 1)

    def forward(self, data):
        x = self.feature_pre(data.x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = global_mean_pool(x, data.batch)

        return self.readout(x).squeeze(-1)
