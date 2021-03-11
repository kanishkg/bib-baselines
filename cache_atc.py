import os
import random
from argparse import ArgumentParser

import h5py as h5py
import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from atc.models import ATCEncoder
from pemirl.datasets import CacheDataset

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--seed', type=int, default=4)

# data specific args
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--types', nargs='+', type=str, default=['co', 'pr'],
                    help='types of tasks used for training / testing')
parser.add_argument('--size', type=int, default=84)

# checkpoint path
parser.add_argument('--ckpt', type=str, default='./')
parser.add_argument('--cache_file', type=str, default='./')

# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

# parse args
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# load model with weights and hparams in eval mode
model = ATCEncoder.load_from_checkpoint(args.ckpt)
model.eval()
model.freeze()

# load train and val datasets and loaders
print("loading train dataset")
train_dataset = CacheDataset(args.data_path, types=args.types, size=(args.size, args.size),
                             mode='train', process_data=args.cache)
print("loading val dataset")
val_dataset = CacheDataset(args.data_path, types=args.types, size=(args.size, args.size),
                           mode='val', process_data=args.cache)

train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=1,
                          pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=1, pin_memory=True)

# cache train dataset
for b, batch in enumerate(train_loader):
    print(f'train {b}/ {len(train_dataset)}')
    frames, actions = batch
    frame_embeddings = model(frames)
    frame_embeddings = frame_embeddings[0, :, :].cpu().numpy()
    actions = actions[0, :].cpu().numpy()
    with h5py.File(f'{os.path.join(args.path, args.cache_file)}_train.h5', 'a') as f:
        f.create_dataset(f'{b}_s', data=frame_embeddings)
        f.create_dataset(f'{b}_a', data=actions)

# cache val dataset
for b, batch in enumerate(val_loader):
    print(f'val {b}/ {len(train_dataset)}')
    frames, actions = batch
    frame_embeddings = model(frames)
    frame_embeddings = frame_embeddings[0, :, :].cpu().numpy()
    actions = actions[0, :].cpu().numpy()
    with h5py.File(f'{os.path.join(args.path, args.cache_file)}_val.h5', 'a') as f:
        f.create_dataset(f'{b}_s', data=frame_embeddings)
        f.create_dataset(f'{b}_a', data=actions)
