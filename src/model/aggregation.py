import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, Set2Set


class GraphAggr(nn.Module):
    def __init__(self, args):
        super(GraphAggr, self).__init__()

        if args.global_aggregation == 'sum':
            self.aggr = global_add_pool
        elif args.global_aggregation == 'mean':
            self.aggr = global_mean_pool
        elif args.global_aggregation == 'set2set':
            self.aggr = Set2Set(args.hidden_size, args.processing_steps)

    def forward(self, x):
        return self.aggr(x)
