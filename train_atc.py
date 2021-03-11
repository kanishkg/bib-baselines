import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from atc.datasets import FrameDataset
from atc.models import ATCEncoder

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--seed', type=int, default=4)

# data specific args
parser.add_argument('--data_path', type=str, default='./data.npy')
parser.add_argument('--cache', type=int, default=0)
parser.add_argument('--types', nargs='+', type=str, default=['co', 'pr'],
                    help='types of tasks used for training / testing')
parser.add_argument('--shift', type=int, default=1)
parser.add_argument('--random_shift', type=int, default=0)
parser.add_argument('--size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)

# add model specific args
parser = ATCEncoder.add_model_specific_args(parser)

# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

model = ATCEncoder(args)

train_dataset = FrameDataset(args.data_path, types=args.types, size=[args.size, args.size],
                             random_shift=args.random_shift, mode='train')
val_dataset = FrameDataset(args.data_path, types=args.types, size=[args.size, args.size],
                           random_shift=args.random_shift, mode='val')


train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = Trainer.from_argparse_args(args)
trainer.fit(model, train_loader, val_loader)
