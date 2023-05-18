import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch.loggers as pl_loggers

import os

from model import LightningModel
from data import load_mnist
from transform import train_transform, val_transform
from utils import read_yaml


# config
config = read_yaml('config.yaml')
print(config)

# load data
train_dataset = load_mnist(root = './data/train', train = True, transform = train_transform())
val_dataset = load_mnist(root = './data/test', train = False, transform = val_transform())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batchsize'], shuffle=True)

# modelcheckpoint
ckpt_callback = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'min',
    dirpath = os.path.join(config['dirname'], config['expname']),
    filename = '{epoch}-{val_loss:.2f}-{val_score:.2f}'
)
# tensorboard logger
tb_logger = pl_loggers.TensorBoardLogger(save_dir = config['dirname'], name = config['expname'])

# load model
model = LightningModel(config)

# trainer
trainer = pl.Trainer(
     max_epochs = config['max_epochs'],
    accelerator = 'auto',
    precision = 16,
    logger = tb_logger,
    callbacks = [ ckpt_callback],
)

trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

