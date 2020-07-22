import os
import math
import torch
import pandas as pd

from features.featurization import construct_loader
from utils import Standardizer, create_logger

from model.gnn import GNN
from model.training import train, eval, test, build_lr_scheduler
from model.parsing import parse_train_args

args = parse_train_args()
torch.manual_seed(args.seed)
logger = create_logger('train', args.log_dir)

train_loader, val_loader = construct_loader(args)
mean = train_loader.dataset.mean
std = train_loader.dataset.std
stdzer = Standardizer(mean, std)

# create model, optimizer, scheduler, and loss fn
model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = build_lr_scheduler(optimizer, args, len(train_loader.dataset))
loss = torch.nn.MSELoss(reduction='sum')
best_val_loss = math.inf
best_epoch = 0

# record args, optimizer, and scheduler info
logger.info('Arguments are...')
for arg in vars(args):
    logger.info(f'{arg}: {getattr(args, arg)}')
logger.info(f'\nOptimizer parameters are:\n{optimizer}\n')
logger.info(f'Scheduler state dict is:')
for key, value in scheduler.state_dict().items():
    logger.info(f'{key}: {value}')
logger.info('')

# multi gpu training
if torch.cuda.device_count() > 1:
    logger.info(f'Using {torch.cuda.device_count()} GPUs for training...')
    model = torch.nn.DataParallel(model)

# train
logger.info("Starting training...")
for epoch in range(0, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, loss, stdzer, args.device, scheduler)
    logger.info(f"Epoch {epoch}: Training Loss {train_loss}")

    val_loss = eval(model, val_loader, loss, stdzer, args.device)
    logger.info(f"Epoch {epoch}: Validation Loss {val_loss}")

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model'))
logger.info(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch}")

# load best model
model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
state_dict = torch.load(os.path.join(args.log_dir, 'best_model'), map_location=args.device)
model.load_state_dict(state_dict)

# predict test data
test_loader = construct_loader(args, modes='test')
preds, test_loss = test(model, test_loader, loss, stdzer, args.device)
logger.info(f"Test Loss {test_loss}")

# save predictions
smiles = test_loader.dataset.smiles
preds_path = os.path.join(args.log_dir, 'preds.csv')
pd.DataFrame(list(zip(smiles, preds)), columns=['smiles', 'prediction']).to_csv(preds_path, index=False)
