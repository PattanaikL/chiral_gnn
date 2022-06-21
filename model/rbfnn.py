import torch
import torch.nn as nn
import model.radial_basis_function as rbf

class RBFNN(nn.Module):
    def __init__(self, args):
        super(RBFNN, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.morgan = args.morgan
        self.bits = args.bits
        self.batch = args.batch_size
        self.add_feature = args.add_feature_path

        if self.morgan:
            if self.add_feature:
                self.input_linear_layer = nn.Linear(args.bits + args.n_add_feature, args.rbfnn_width)
            else:
                self.input_linear_layer = nn.Linear(args.bits, args.rbfnn_width)
        else:
            if self.add_feature:
                self.input_linear_layer = nn.Linear(args.hidden_size + args.n_add_feature, args.rbfnn_width)
            else:
                self.input_linear_layer = nn.Linear(args.hidden_size, args.rbfnn_width)

        for i in range(args.rbfnn_depth):
            self.rbf_layers.append(rbf.RBF(args.rbfnn_width, args.rbfnn_centers, rbf.gaussian)) #TO DO : Make general
            self.linear_layers.append(nn.Linear(args.rbfnn_centers, args.rbfnn_width))
        self.output_linear_layer = nn.Linear(args.rbfnn_width, args.n_out)

    def forward(self, x):
        if self.morgan:
            out = x.x
            out = out.reshape([int(len(x.x)/self.bits), self.bits]) #TO DO: Dataset batch size not accurate hence done janky
            if self.add_feature:
                out = torch.cat((out, x.add_feature), dim=1)
        else:
            out = x
        out = self.input_linear_layer(out)
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        out = self.output_linear_layer(out)
        return out
