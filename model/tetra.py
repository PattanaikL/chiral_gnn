import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TetraPermuter(nn.Module):

    def __init__(self, hidden, device):
        super(TetraPermuter, self).__init__()
        
        self.W_bs = nn.ModuleList([copy.deepcopy(nn.Linear(hidden, hidden)) for _ in range(4)])
        self.device = device
        self.drop = nn.Dropout(p=0.2)
        self.reset_parameters()
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))
            
        self.perms = dict()
        self.perms["mono"] = torch.tensor([[0, 0, 0, 0]])
        self.perms["di"] = torch.tensor([[0, 0, 1, 1],
                                         [0, 1, 0, 1],
                                         [0, 1, 1, 0],
                                         [1, 0, 0, 1],
                                         [1, 0, 1, 0],
                                         [1, 1, 0, 0]])
        self.perms["tri"] = torch.tensor([[0, 0, 1, 2],
                                        [0, 0, 2, 1],
                                        [0, 1, 0, 2],
                                        [0, 1, 2, 0],
                                        [0, 2, 0, 1],
                                        [0, 2, 1, 0],
                                        [1, 0, 1, 2],
                                        [1, 0, 2, 1],
                                        [1, 1, 0, 2],
                                        [1, 1, 2, 0],
                                        [1, 2, 0, 1],
                                        [1, 2, 1, 0],
                                        [2, 0, 1, 2],
                                        [2, 0, 2, 1],
                                        [2, 1, 0, 2],
                                        [2, 1, 2, 0],
                                        [2, 2, 0, 1],
                                        [2, 2, 1, 0]])
        self.perms["tetra_chiral"] = torch.tensor([[0, 1, 2, 3],
                                                 [0, 2, 3, 1],
                                                 [0, 3, 1, 2],
                                                 [1, 0, 3, 2],
                                                 [1, 2, 0, 3],
                                                 [1, 3, 2, 0],
                                                 [2, 0, 1, 3],
                                                 [2, 1, 3, 0],
                                                 [2, 3, 0, 1],
                                                 [3, 0, 2, 1],
                                                 [3, 1, 0, 2],
                                                 [3, 2, 1, 0]])
        self.perms["tetra_nonchiral"] = torch.tensor([[0, 1, 2, 3],
                                                    [0, 1, 3, 2],
                                                    [0, 2, 1, 3],
                                                    [0, 2, 3, 1],
                                                    [0, 3, 1, 2],
                                                    [0, 3, 2, 1],
                                                    [1, 0, 2, 3],
                                                    [1, 0, 3, 2],
                                                    [1, 2, 3, 0],
                                                    [1, 2, 0, 3],
                                                    [1, 3, 0, 2],
                                                    [1, 3, 2, 0],
                                                    [2, 0, 1, 3],
                                                    [2, 0, 3, 1],
                                                    [2, 1, 0, 3],
                                                    [2, 1, 3, 0],
                                                    [2, 3, 0, 1],
                                                    [2, 3, 1, 0],
                                                    [3, 0, 1, 2],
                                                    [3, 0, 2, 1],
                                                    [3, 1, 0, 2],
                                                    [3, 1, 2, 0],
                                                    [3, 2, 0, 1],
                                                    [3, 2, 1, 0]])

    def reset_parameters(self):
        gain = 0.5
        for W_b in self.W_bs:
            nn.init.xavier_uniform_(W_b.weight, gain=gain)
            gain += 0.5

    def forward(self, x, n_neighbor):
        nei_messages = torch.zeros([x.size(0), x.size(2)]).to(self.device)

        for p in self.perms[n_neighbor]:
            nei_messages_list = [self.drop(F.tanh(l(t))) for l, t in zip(self.W_bs, torch.split(x[:, p, :], 1, dim=1))]
            nei_messages += self.drop(F.relu(torch.cat(nei_messages_list, dim=1).sum(dim=1)))
        return self.mlp_out(nei_messages / 3.)


class ConcatTetraPermuter(nn.Module):

    def __init__(self, hidden, device):
        super(ConcatTetraPermuter, self).__init__()

        self.n_nums = dict()
        self.n_nums["mono"] = 1
        self.n_nums["di"] = 2
        self.n_nums["tri"] = 3
        self.n_nums["tetra_nonchiral"] = 4
        self.n_nums["tetra_chiral"] = 4
        
        self.W_bs = nn.Linear(hidden*4, hidden)
        torch.nn.init.xavier_normal_(self.W_bs.weight, gain=1.0)
        self.hidden = hidden
        self.device = device
        self.drop = nn.Dropout(p=0.2)
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

        self.perms = dict()
        self.perms["mono"] = torch.tensor([[0, 0, 0, 0]])
        self.perms["di"] = torch.tensor([[0, 0, 1, 1],
                                         [0, 1, 0, 1],
                                         [0, 1, 1, 0],
                                         [1, 0, 0, 1],
                                         [1, 0, 1, 0],
                                         [1, 1, 0, 0]])
        self.perms["tri"] = torch.tensor([[0, 0, 1, 2],
                                        [0, 0, 2, 1],
                                        [0, 1, 0, 2],
                                        [0, 1, 2, 0],
                                        [0, 2, 0, 1],
                                        [0, 2, 1, 0],
                                        [1, 0, 1, 2],
                                        [1, 0, 2, 1],
                                        [1, 1, 0, 2],
                                        [1, 1, 2, 0],
                                        [1, 2, 0, 1],
                                        [1, 2, 1, 0],
                                        [2, 0, 1, 2],
                                        [2, 0, 2, 1],
                                        [2, 1, 0, 2],
                                        [2, 1, 2, 0],
                                        [2, 2, 0, 1],
                                        [2, 2, 1, 0]])
        self.perms["tetra_chiral"] = torch.tensor([[0, 1, 2, 3],
                                                 [0, 2, 3, 1],
                                                 [0, 3, 1, 2],
                                                 [1, 0, 3, 2],
                                                 [1, 2, 0, 3],
                                                 [1, 3, 2, 0],
                                                 [2, 0, 1, 3],
                                                 [2, 1, 3, 0],
                                                 [2, 3, 0, 1],
                                                 [3, 0, 2, 1],
                                                 [3, 1, 0, 2],
                                                 [3, 2, 1, 0]])
        self.perms["tetra_nonchiral"] = torch.tensor([[0, 1, 2, 3],
                                                    [0, 1, 3, 2],
                                                    [0, 2, 1, 3],
                                                    [0, 2, 3, 1],
                                                    [0, 3, 1, 2],
                                                    [0, 3, 2, 1],
                                                    [1, 0, 2, 3],
                                                    [1, 0, 3, 2],
                                                    [1, 2, 3, 0],
                                                    [1, 2, 0, 3],
                                                    [1, 3, 0, 2],
                                                    [1, 3, 2, 0],
                                                    [2, 0, 1, 3],
                                                    [2, 0, 3, 1],
                                                    [2, 1, 0, 3],
                                                    [2, 1, 3, 0],
                                                    [2, 3, 0, 1],
                                                    [2, 3, 1, 0],
                                                    [3, 0, 1, 2],
                                                    [3, 0, 2, 1],
                                                    [3, 1, 0, 2],
                                                    [3, 1, 2, 0],
                                                    [3, 2, 0, 1],
                                                    [3, 2, 1, 0]])

    def forward(self, x, n_neighbor):

        nei_messages = torch.zeros([x.size(0), x.size(2)]).to(self.device)

        for p in self.perms[n_neighbor]:
            nei_messages += self.drop(F.relu(self.W_bs(x[:, p, :].view(x.size(0), self.hidden*4))))
        return self.mlp_out(nei_messages / 3.)


class TetraDifferencesProduct(nn.Module):

    def __init__(self, hidden):
        super(TetraDifferencesProduct, self).__init__()

        self.n_nums = dict()
        self.n_nums["mono"] = 1
        self.n_nums["di"] = 2
        self.n_nums["tri"] = 3
        self.n_nums["tetra_nonchiral"] = 4
        self.n_nums["tetra_chiral"] = 4
        
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

    def forward(self, x, n_neighbor):

        num = self.n_nums[n_neighbor]
        indices = torch.arange(num).to(x.device)
        message_n_nbs = [x.index_select(dim=1, index=i).squeeze(1) for i in indices]
        message_n = torch.ones_like(message_n_nbs[0])

        # note: this will zero out reps for chiral centers with multiple carbon neighbors on first pass
        for i in range(num):
            for j in range(i + 1, num):
                message_n = torch.mul(message_n, (message_n_nbs[i] - message_n_nbs[j]))
        message_n = torch.sign(message_n) * torch.pow(torch.abs(message_n) + 1e-6, 1 / 6)
        return self.mlp_out(message_n)


def get_tetra_update(args):

    if args.message == 'tetra_permute':
        return TetraPermuter(args.hidden_size, args.device)
    elif args.message == 'tetra_permute_concat':
        return ConcatTetraPermuter(args.hidden_size, args.device)
    elif args.message == 'tetra_pd':
        return TetraDifferencesProduct(args.hidden_size)
    else:
        raise ValueError("Invalid message type.")
