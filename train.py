import math
import torch

from features.featurization import construct_loader
from utils import Standardizer, create_logger

from model.gnn import GNN
from model.training import train, test, build_lr_scheduler
from model.parsing import parse_train_args

args = parse_train_args()
torch.manual_seed(args.seed)
logger = create_logger('train', args.log_dir)

train_loader, val_loader = construct_loader(args)
mean = train_loader.dataset.mean
std = train_loader.dataset.std
stdzer = Standardizer(mean, std)

model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                        factor=0.7, patience=5,
#                                                        min_lr=0.00001)
scheduler = build_lr_scheduler(optimizer, args, len(train_loader.dataset))
loss = torch.nn.MSELoss(reduction='sum')
best_val_loss = math.inf
best_epoch = 0

logger.info("Starting training...")
for epoch in range(1, args.n_epochs):
    train_loss = train(model, train_loader, optimizer, loss, stdzer, args.device, scheduler)
    logger.info("Epoch {}: Training Loss {}".format(epoch, train_loss))

    val_loss = test(model, val_loader, loss, stdzer, args.device)
    logger.info("Epoch {}: Validation Loss {}".format(epoch, val_loss))

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        # torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model'))

logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
