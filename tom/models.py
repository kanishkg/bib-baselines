from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from atc.models import MlpModel, ATCEncoder
from atc.utils import update_state_dict
from tom.datasets import *

class BCMLP(pl.LightningModule):
    """
    Implentation of the Behavior cloning algorithm with 
    an MLP to encode the familiarization trials. 
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--state_dim', type=int, default=128)
        parser.add_argument('--action_dim', type=int, default=2)
        parser.add_argument('--beta', type=float, default=0.01)
        parser.add_argument('--gamma', type=float, default=0.0)
        parser.add_argument('--context_dim', nargs='+', type=int, default=32)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--size', type=int, default=84)
        parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 32, 32])
        parser.add_argument('--filter', nargs='+', type=int, default=[3, 3, 3, 3])
        parser.add_argument('--strides', nargs='+', type=int, default=[2, 2, 2, 1])
        parser.add_argument('--process_data', type=int, default=0)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.lr = self.hparams.lr
        self.state_dim = self.hparams.state_dim
        self.action_dim = self.hparams.action_dim
        self.context_dim = self.hparams.context_dim
        self.beta = self.hparams.beta
        self.gamma = self.hparams.gamma
        self.dropout = self.hparams.dropout

        # load the pretrained encoder
        self.encoder = ATCEncoder.load_from_checkpoint(
            '/data/kvg245/bib-tom/lightning_logs/version_911764/checkpoints/epoch=31-step=342118.ckpt')
        self.context_enc_mean = MlpModel(self.state_dim + self.action_dim, hidden_sizes=[64, 64],
                                         output_size=self.context_dim)
        self.policy = MlpModel(input_size=self.state_dim + self.context_dim, hidden_sizes=[256, 128, 256],
                               output_size=self.action_dim, dropout=self.dropout)

        self.past_samples = []

    def forward(self, batch):
        dem_frames, dem_actions, test_frames, test_actions = batch
        dem_frames = dem_frames.float()
        dem_actions = dem_actions.float()
        test_actions = test_actions.float()
        test_frames = test_frames.float()

        dem_states, _ = self.encoder.encoder(dem_frames)
        test_states, _ = self.encoder.encoder(test_frames)
        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        context_mean_samples = self.context_enc_mean(dem_traj)
        context = torch.mean(context_mean_samples, dim=1)

        # concat context embedding to the state embedding of test trajectory
        test_context_states = torch.cat([context.unsqueeze(1), test_states], dim=2)
        b, s, d = test_context_states.size()
        test_context_states = test_context_states.view(b * s, d)
        test_actions = test_actions.view(b * s, -1)

        # for each state in the test states calculate action
        test_actions_pred = F.tanh(self.policy(test_context_states))

        return test_actions, test_actions_pred

    def training_step(self, batch, batch_idx):
        test_actions, test_actions_pred = self.forward(batch)
        loss = F.mse_loss(test_actions, test_actions_pred)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        test_actions, test_actions_pred = self.forward(batch)
        loss = F.mse_loss(test_actions, test_actions_pred)
        self.log('val_loss', loss, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def train_dataloader(self):
        train_dataset = TransitionDataset(self.hparams.data_path, types=self.hparams.types, mode='train',
                                             process_data=self.hparams.process_data,
                                             size=(self.hparams.size, self.hparams.size))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size,
                                  num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_datasets = []
        val_loaders = []
        for t in self.hparams.types:
            val_datasets.append(TransitionDataset(self.hparams.data_path, types=[t], mode='val',
                                                     process_data=self.hparams.process_data,
                                                     size=(self.hparams.size, self.hparams.size)))
            val_loaders.append(DataLoader(dataset=val_datasets[-1], batch_size=self.hparams.batch_size,
                                          num_workers=self.hparams.num_workers, pin_memory=True, shuffle=False))
        return val_loaders

class BCRNN(pl.LightningModule):
    """
    Implementation of the baseline model for the BC-RNN algorithm.
    An LSTM is used to encode the familiarization trials
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--state_dim', type=int, default=128)
        parser.add_argument('--action_dim', type=int, default=2)
        parser.add_argument('--beta', type=float, default=0.01)
        parser.add_argument('--gamma', type=float, default=0.0)
        parser.add_argument('--context_dim', nargs='+', type=int, default=32)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--size', type=int, default=84)
        parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 32, 32])
        parser.add_argument('--filter', nargs='+', type=int, default=[3, 3, 3, 3])
        parser.add_argument('--strides', nargs='+', type=int, default=[2, 2, 2, 1])
        parser.add_argument('--process_data', type=int, default=0)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.lr = self.hparams.lr
        self.state_dim = self.hparams.state_dim
        self.action_dim = self.hparams.action_dim
        self.context_dim = self.hparams.context_dim
        self.beta = self.hparams.beta
        self.gamma = self.hparams.gamma
        self.dropout = self.hparams.dropout

        self.encoder = ATCEncoder.load_from_checkpoint(
            '/data/kvg245/bib-tom/lightning_logs/version_911764/checkpoints/epoch=31-step=342118.ckpt')
        self.context_enc = torch.nn.LSTM(self.state_dim + self.action_dim, self.context_dim, 2, batch_first=True,
                                         bidirectional=True)

        self.policy = MlpModel(input_size=self.state_dim + self.context_dim*2, hidden_sizes=[256, 128, 256],
                               output_size=self.action_dim, dropout=self.dropout)
        self.past_samples = []

    def forward(self, batch):
        dem_frames, dem_actions, dem_lens, query_frame, target_action = batch
        dem_frames = dem_frames.float()
        dem_actions = dem_actions.float()
        target_action = target_action.float()
        query_frame = query_frame.float()

        # flatten and encode all dem frames
        b, l, s, c, h, w = dem_frames.size()
        dem_frames = dem_frames.view(b * l * s, c, h, w)
        dem_states, _ = self.encoder.encoder(dem_frames)
        # reshape to batch, seq, state_dim
        dem_states = dem_states.view(b * l, s, -1)
        dem_actions = dem_actions.view(b * l, s, -1)
        dem_lens = torch.tensor([t for dl in dem_lens for t in dl]).to(torch.int64).cpu()
        # concatenate states and actions to get expert trajectory
        dem_traj = torch.cat([dem_states, dem_actions], dim=2)

        # embed expert trajectory to get a context embedding batch x samples x dim
        dem_lens = dem_lens.view(-1)
        x_lstm = torch.nn.utils.rnn.pack_padded_sequence(dem_traj, dem_lens, batch_first=True, enforce_sorted=False)
        x_lstm, _ = self.context_enc(x_lstm)
        x_lstm, _ = torch.nn.utils.rnn.pad_packed_sequence(x_lstm, batch_first=True)
        x_out = x_lstm[:, -1, :]
        x_out = x_out.view(b, l, -1)
        context = torch.mean(x_out, dim=1)

        # concat context embedding to the state embedding of test trajectory
        test_states, _ = self.encoder.encoder(query_frame)
        test_context_states = torch.cat([context, test_states], dim=1)

        # for each state in the test states calculate action
        test_actions_pred = F.tanh(self.policy(test_context_states))

        return target_action, test_actions_pred

    def training_step(self, batch, batch_idx):
        test_actions, test_actions_pred = self.forward(batch)
        loss = F.mse_loss(test_actions, test_actions_pred)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        test_actions, test_actions_pred = self.forward(batch)
        loss = F.mse_loss(test_actions, test_actions_pred)
        self.log('val_loss', loss, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def train_dataloader(self):
        train_dataset = TransitionDatasetSequence(self.hparams.data_path, types=self.hparams.types, mode='train',
                                             process_data=self.hparams.process_data,
                                             size=(self.hparams.size, self.hparams.size))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.hparams.batch_size,
                                  collate_fn=collate_function_seq, num_workers=self.hparams.num_workers,
                                  pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_datasets = []
        val_loaders = []
        for t in self.hparams.types:
            val_datasets.append(TransitionDatasetSequence(self.hparams.data_path, types=[t], mode='val',
                                                     process_data=self.hparams.process_data,
                                                     size=(self.hparams.size, self.hparams.size)))
            val_loaders.append(DataLoader(dataset=val_datasets[-1], batch_size=self.hparams.batch_size,
                                          collate_fn=collate_function_seq, num_workers=self.hparams.num_workers,
                                          pin_memory=True, shuffle=False))
        return val_loaders
    