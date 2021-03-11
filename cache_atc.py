import random
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer

from atc.models import ATCEncoder

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--seed', type=int, default=4)

# data specific args
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--types', nargs='+', type=str, default=['co', 'pr'],
                    help='types of tasks used for training / testing')
parser.add_argument('--size', type=int, default=84)

# saved model
parser.add_argument('--ckpt', type=str, default='./')

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


