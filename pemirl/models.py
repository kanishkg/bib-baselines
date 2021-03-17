import copy
from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
import pytorch_lightning as pl

from atc.models import MlpModel, ATCEncoder


class PEMIRL(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--latent_size', type=int, default=128)
        parser.add_argument('--anchor_size', type=int, default=256)
        parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 32, 32])
        parser.add_argument('--filter', nargs='+', type=int, default=[3, 3, 3, 3])
        parser.add_argument('--strides', nargs='+', type=int, default=[2, 2, 2, 1])
        parser.add_argument('--target_update_interval', type=int, default=1)
        parser.add_argument('--target_update_tau', type=float, default=0.01)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.lr = hparams.lr
        self.state_dim = hparams.state_dim
        self.action_dim = hparams.action_dim
        self.context_dim = hparams.action_dim
        self.gamma = hparams.gamma

        self.context_enc_mean = nn.LSTM(self.state_dim + self.action_dim, self.state_dim, 1, batch_first=True,
                                        bidirectional=False)
        self.context_enc_std = nn.LSTM(self.state_dim + self.action_dim, self.state_dim, 1, batch_first=True,
                                       bidirectional=False)

        self.policy = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[64, 64],
                               output_size=self.action_dim)

        self.r = MlpModel(input_size=self.state_dim + self.context_dim + self.action_dim, hidden_sizes=[64, 32],
                          output_size=1)

        self.linfo = nn.CrossEntropyLoss(reduction='none')

        self.past_samples = []

    def training_step(self, batch, batch_idx):
        dem_states, dem_actions, test_states, test_actions, dem_l, test_l, test_mask = batch

        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding
        x_states = torch.nn.utils.rnn.pack_padded_sequence(dem_traj, dem_l, batch_first=True, enforce_sorted=False)
        x_mean, _ = self.context_enc_mean(x_states)
        x_mean, _ = torch.nn.utils.rnn.pad_packed_sequence(x_mean, batch_first=True)
        x_std, _ = self.context_enc_std(x_states)
        x_std, _ = torch.nn.utils.rnn.pad_packed_sequence(x_std, batch_first=True)

        context_mean = x_mean[:, -1, :]
        context_std = x_std[:, -1, :]
        context = torch.normal(context_mean, context_std)

        # tile to get vector of size b x len x dim
        context_tiled = context.unsqueeze(1).repeat(1, test_states.size(1), 1)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context_tiled, test_states], dim=2)

        # for each state in the test states calculate action
        b, l, d = test_context_states.size()
        test_context_states = test_context_states.view(b * l, d)

        # calculate the reward for each state that is observed
        test_reward = self.r(test_context_states)
        test_reward = test_reward.view(b, l, 1).sum(1)




        test_action_pred = self.policy(test_cat_states)
        test_action_pred_states_context
        test_action_pred = test_action_pred.view(b, l, -1)

        ce_loss = self.linfo(test_actions, test_action_pred)
        ce_loss.masked = ce_loss * test_mask
