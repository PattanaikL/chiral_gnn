import os
import math
import torch
import pandas as pd

from features.featurization import construct_loader
from utils import Standardizer, create_logger, get_loss_func

from model.gnn import GNN
from model.training import train, eval, test, build_lr_scheduler
from model.parsing import parse_train_args

args = parse_train_args()
torch.manual_seed(args.seed)
logger = create_logger('train', args.log_dir)

train_loader, val_loader = construct_loader(args)
mean = train_loader.dataset.mean
std = train_loader.dataset.std
stdzer = Standardizer(mean, std, args.task)

# create model, optimizer, scheduler, and loss fn
model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = build_lr_scheduler(optimizer, args, len(train_loader.dataset))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=args.lr/100)
loss = get_loss_func(args)
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

# train
logger.info("Starting training...")
for epoch in range(0, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, loss, stdzer, scheduler, args)
    logger.info(f"Epoch {epoch}: Training Loss {train_loss}")

    val_loss = eval(model, val_loader, loss, stdzer, args)
    scheduler.step(val_loss)
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
preds, test_loss = test(model, test_loader, loss, stdzer, args)
logger.info(f"Test Loss {test_loss}")

# save predictions
mol_blocks = test_loader.dataset.mol_blocks
preds_path = os.path.join(args.log_dir, 'preds.csv')
pd.DataFrame(list(zip(mol_blocks, preds)), columns=['mol_block', 'prediction']).to_csv(preds_path, index=False)
