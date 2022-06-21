from argparse import ArgumentParser, Namespace
import torch


def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--split_path', type=str,
                        help='Path to .npy file containing train/val/test split indices')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--task', type=str, default='regression',
                        help='Regression or classification task')
    parser.add_argument('--pytorch_seed', type=int, default=0,
                        help='Seed for PyTorch randomness (e.g., random initial weights).')
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=60,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Max Learning rate, inital learning rate is 1/10 max and final lr = inital lr')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of workers to use in dataloader')
    parser.add_argument('--no_shuffle', action='store_true', default=False,
                        help='Whether or not to retain default ordering during training')
    parser.add_argument('--shuffle_pairs', action='store_true', default=False,
                        help='Whether or not to shuffle only pairs of stereoisomers')
    parser.add_argument('--additional_atom_feat_path', type=str,
                        help='Path to pickle file containing a pandas dataframe')

    # Model arguments
    parser.add_argument('--gnn_type', type=str,
                        choices=['gin', 'gcn', 'dmpnn', 'gat', 'gatv2'],
                        help='Type of gnn to use')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability')
    parser.add_argument('--graph_pool', type=str, default='sum',
                        choices=['sum', 'mean', 'max', 'attn', 'set2set'],
                        help='How to aggregate atom representations to molecule representation')
    parser.add_argument('--message', type=str, default='sum',
                        choices=['sum', 'tetra_pd', 'tetra_permute', 'tetra_permute_concat'],
                        help='How to pass neighbor messages')
    parser.add_argument('--chiral_features', action='store_true', default=False,
                        help='Use local chiral atom features')
    parser.add_argument('--global_chiral_features', action='store_true', default=False,
                        help='Use global chiral atom features')
    parser.add_argument('--ffn_depth', type=int, default=5, help='FFN layers less the linear output and input layer')
    parser.add_argument('--ffn_hidden_size', type=int, default=300, help='Dimensionality of hidden layers in FFN')
    parser.add_argument('--add_feature_path', type=str,
                        help='Path to csv file containing rdkit features')
    parser.add_argument('--n_add_feature', type=int,
                        help='Number of rdkit features per smiles')
    parser.add_argument('--n_out', type=int, default=1,
                        help='Number of ouputs in a multi-target problem')
    parser.add_argument('--scaled_err', default=True)
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--n_fold', type=int, default=1)
    parser.add_argument('--gat_head', type=int, default=1)
    parser.add_argument('--dense', type=bool, default=False)
    parser.add_argument('--target_weights', nargs="+", type=float , default=None)
    parser.add_argument('--pre_train_path', type=str, default=None)

    #Argument for RBFNN mode - GNN - RBFNN
    parser.add_argument('--rbfnn', type=bool, default=False)
    parser.add_argument('--rbfnn_depth', type=int, default=1)
    parser.add_argument('--rbfnn_width', type=int, default=9)
    parser.add_argument('--rbfnn_centers', type=int, default=40)
    parser.add_argument('--rbfnn_basis_func', type=str, default='gaussian',
                        choices=['gaussian', 'linear', 'quadratic', 'inverse_quadratic', 'multiquadric',
                                 'inverse_multiquadric', 'spline', 'poisson_one', 'poisson_two', 'matern32', 'matern52'])
    #Arguments for Morgan fingerprint
    parser.add_argument('--morgan', type=bool, default=False)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--bits', type=int, default=1024)


def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    if args.message.startswith('tetra'):
        setattr(args, 'tetra', True)
    else:
        setattr(args, 'tetra', False)

    # shuffle=False for custom sampler
    if args.shuffle_pairs:
        setattr(args, 'no_shuffle', True)
    setattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args
