import os
import random
from argparse import ArgumentParser

import pickle
import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from atc.models import ATCEncoder
from irl.datasets import CacheDataset

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--seed', type=int, default=4)

# data specific args
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--types', nargs='+', type=str, default=['co', 'pr'],
                    help='types of tasks used for training / testing')
parser.add_argument('--size', type=int, default=84)
parser.add_argument('--cache', type=int, default=0)

# checkpoint path
parser.add_argument('--ckpt', type=str, default='./')
parser.add_argument('--hparams', type=str, default='./')
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
model = ATCEncoder.load_from_checkpoint(args.ckpt, hparams_file=args.hparams)
model.eval()
model.freeze()

# load train and val datasets and loaders
print("loading train dataset")
train_dataset = CacheDataset(args.data_path, types=args.types, size=(args.size, args.size),
                             mode='train', process_data=args.cache)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=1,
                          pin_memory=True, shuffle=False)

print("loading val dataset")
val_datasets = []
val_loaders = []
args.types = sorted(args.types)
print(args.types)
for t in args.types:
    val_datasets.append(CacheDataset(args.data_path, types=[t], size=(args.size, args.size),
                                     mode='val', process_data=args.cache))

    val_loaders.append(
        DataLoader(dataset=val_datasets[-1], batch_size=1, num_workers=1, pin_memory=True, shuffle=False))

# cache train dataset
store_dict = {}
for b, batch in enumerate(train_loader):
    print(f'train {b}/ {len(train_dataset)}')
    frames, actions = batch
    frame_embeddings = model(frames)
    frame_embeddings = frame_embeddings[0, :, :].cpu().numpy()
    actions = actions[0, :].cpu().numpy()
    store_dict[f'{b}_s'] = frame_embeddings
    store_dict[f'{b}_a'] = actions

type_str = '_'.join(args.types)
with open(f'{os.path.join(args.data_path, args.cache_file)}_train_{type_str}.pickle', 'wb') as handle:
    pickle.dump(store_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# cache val dataset
for v, t in zip(val_loaders, args.types):
    store_dict = {}
    for b, batch in enumerate(v):
        print(f'val {b}')
        frames, actions = batch
        frame_embeddings = model(frames)
        frame_embeddings = frame_embeddings[0, :, :].cpu().numpy()
        actions = actions[0, :].cpu().numpy()
        store_dict[f'{b}_s'] = frame_embeddings
        store_dict[f'{b}_a'] = actions
    with open(f'{os.path.join(args.data_path, args.cache_file)}_val_{t}.pickle', 'wb') as handle:
        pickle.dump(store_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
