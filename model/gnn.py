import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from .layers import GCNConv, GINEConv, DMPNNConv, get_tetra_update
from torch_geometric.nn import GATConv


class GNN(nn.Module):
    def __init__(self, args, num_node_features, num_edge_features):
        super(GNN, self).__init__()

        self.depth = args.depth
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.gnn_type = args.gnn_type
        self.graph_pool = args.graph_pool
        self.tetra = args.tetra
        self.task = args.task
        self.ffn_depth = args.ffn_depth
        self.ffn_hidden_size = args.ffn_hidden_size
        self.add_feature = args.add_feature_path
        self.n_add_feature = args.n_add_feature
        self.n_out = args.n_out

        if self.gnn_type == 'dmpnn':
            self.edge_init = nn.Linear(num_node_features + num_edge_features, self.hidden_size)
            self.edge_to_node = DMPNNConv(args)
        else:
            self.node_init = nn.Linear(num_node_features, self.hidden_size)
            self.edge_init = nn.Linear(num_edge_features, self.hidden_size)

        # layers
        self.convs = torch.nn.ModuleList()

        for _ in range(self.depth):
            if self.gnn_type == 'gin':
                self.convs.append(GINEConv(args))
            elif self.gnn_type == 'gcn':
                self.convs.append(GCNConv(args))
            elif self.gnn_type == 'dmpnn':
                self.convs.append(DMPNNConv(args))
            elif self.gnn_type == 'gat':
                self.convs.append(GATConv(args.hidden_size, args.hidden_size))
            else:
                ValueError('Undefined GNN type called {}'.format(self.gnn_type))
        if self.add_feature:
            self.ffn_1 = nn.Linear(self.hidden_size + self.n_add_feature, self.ffn_hidden_size)
        else:
            self.ffn_1 = nn.Linear(self.hidden_size, self.ffn_hidden_size)

        self.ffn = torch.nn.ModuleList()
        for _ in range(self.ffn_depth):
            self.ffn.append(nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size))

        # graph pooling
        if self.tetra:
            self.tetra_update = get_tetra_update(args)

        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool = global_mean_pool
        elif self.graph_pool == "max":
            self.pool = global_max_pool
        elif self.graph_pool == "attn":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                                            torch.nn.BatchNorm1d(2 * self.hidden_size),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * self.hidden_size, 1)))
        elif self.graph_pool == "set2set":
            self.pool = Set2Set(self.hidden_size, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # ffn
        self.mult = 2 if self.graph_pool == "set2set" else 1
        self.ffn_out = nn.Linear(self.ffn_hidden_size, self.n_out) #Change here for multiple outputs

    def forward(self, data):
        x, edge_index, edge_attr, batch, parity_atoms, add_feature = data.x, data.edge_index, data.edge_attr, data.batch, data.parity_atoms, data.add_feature

        if self.gnn_type == 'dmpnn':
            row, col = edge_index
            edge_attr = torch.cat([x[row], edge_attr], dim=1)
            edge_attr = F.relu(self.edge_init(edge_attr))
        else:
            x = F.relu(self.node_init(x))
            edge_attr = F.relu(self.edge_init(edge_attr))

        x_list = [x]
        edge_attr_list = [edge_attr]

        # convolutions
        for l in range(self.depth):
            if self.gnn_type == 'gat':
                x_h = self.convs[l](x_list[-1], edge_index)
            else:
                x_h, edge_attr_h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1], parity_atoms)

            h = edge_attr_h if self.gnn_type == 'dmpnn' else x_h

            if l == self.depth - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            if self.gnn_type == 'dmpnn':
                h += edge_attr_h
                edge_attr_list.append(h)
            else:
                h += x_h
                x_list.append(h)

        # dmpnn edge -> node aggregation
        if self.gnn_type == 'dmpnn':
            h, _ = self.edge_to_node(x_list[-1], edge_index, h, parity_atoms)

        h_ffn = self.pool(h, batch)

        if self.add_feature:
            h_ffn = torch.cat((h_ffn, add_feature), dim=1)

        h_ffn = F.relu(self.ffn_1(h_ffn))
        for l in range(self.ffn_depth):
            h_ffn = self.ffn[l](h_ffn)
            h_ffn = F.relu(h_ffn)

        if self.task == 'regression':
            return self.ffn_out(h_ffn) #.squeeze(-1)#Need to change squeeze
        elif self.task == 'classification':
            return torch.sigmoid(self.ffn(self.pool(h, batch))).squeeze(-1)
