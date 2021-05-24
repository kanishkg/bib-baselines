import random
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer

from irl.models import ContextImitation, ContextImitationPixel, ContextNLL, ContextAIL, OfflineRL, ContextImitationLSTM

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--seed', type=int, default=4)

# data specific args
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--types', nargs='+', type=str, default=['co', 'pr'],
                    help='types of tasks used for training / testing')
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=256)

# add model specific args
parser = ContextImitationLSTM.add_model_specific_args(parser)

# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

# parse args
args = parser.parse_args()
args.types = sorted(args.types)
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# init model
model = ContextImitationLSTM(args)
torch.autograd.set_detect_anomaly(True)
# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = Trainer.from_argparse_args(args)

if args.train:
    trainer.fit(model)
else:
    trainer.test(model)