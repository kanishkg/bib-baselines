import random
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


from tom.models import BCRNN, BCMLP

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--seed', type=int, default=4)

# data specific args
parser.add_argument('--data_path', type=str, default='./')
parser.add_argument('--types', nargs='+', type=str, default=['preference', 'multi_agent', 'single_object', 'instrumental_action'],
                    help='types of tasks used for training / validation')
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model_type', type=str, default='bcmlp')


# add model specific args
parser = BCRNN.add_model_specific_args(parser)
parser = BCMLP.add_model_specific_args(parser)


# add all the available trainer options to argparse
parser = Trainer.add_argparse_args(parser)

# parse args
args = parser.parse_args()
args.types = sorted(args.types)
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
checkpoint_callback = ModelCheckpoint(
    monitor='prior_mse',
    save_top_k=-1,
)
# init model
if args.model_type == 'bcmlp':
    model = BCMLP(args)
elif args.model_type == 'bcrnn':
    model = BCRNN(args)
else:
    raise NotImplementedError
torch.autograd.set_detect_anomaly(True)
# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = Trainer.from_argparse_args(args)
trainer.callbacks = [checkpoint_callback]

if args.train:
    trainer.fit(model)
else:
    raise NotImplementedError